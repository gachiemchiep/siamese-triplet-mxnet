import argparse, time, logging
import os
import mxnet as mx
from tqdm import tqdm
from mxnet import gluon
from mxnet import autograd as ag

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from mxboard import SummaryWriter
from model_zoo.siamese import TripletNet
from loss import TripletSemiHardLoss
from utils import get_transform
from dataloader import BalanceBatchSampler

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--n-classes', type=int, default=8, # each batch contains 10 samples
                        help='Number of classes inside balanced batch.')
    parser.add_argument('--n-samples', type=int, default=8, # each batch contains 10 samples
                        help='Number of sample per class of balanced batch).')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='cifar_resnet20_v2',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('--dataset', type=str, default='cifar10', help="Which dataset to use: cifar10 or cifar100")
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=40,
                        help='number of training epochs.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--save-period', type=int, default=1,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='snapshots',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
    batch_size = opt.n_classes * opt.n_samples
    classes = 10

    dataset = opt.dataset
    if dataset == 'cifar10':
        dataset_train_base = gluon.data.vision.CIFAR10(train=True)
        dataset_test = gluon.data.vision.CIFAR10(train=False)
    elif dataset == 'cifar100':
        dataset_train_base = gluon.data.vision.CIFAR100(train=True, fine_label=True)
        dataset_test = gluon.data.vision.CIFAR100(train=False, fine_label=True)
    else:
        print("Dataset: {} is unknow".format(dataset))

    transform_train, transform_test = get_transform()

    labels = dataset_train_base._label
    batch_sampler = BalanceBatchSampler(labels=labels, n_classes=opt.n_classes, n_samples=opt.n_samples, last_batch='discard')

    triplet_dataset_train_loader = gluon.data.DataLoader(dataset_train_base.transform_first(transform_train), batch_sampler=batch_sampler, num_workers=opt.num_workers)
    dataset_test_loader = gluon.data.DataLoader(dataset_test.transform_first(transform_test), batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)

    train_sample_num = len(dataset_train_base)
    print("Number of train sample: {}".format(train_sample_num))
    print("Number of val sample: {}".format(len(dataset_test)))

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes, 'drop_rate': opt.drop_rate, 'pretrained': True, 'ctx': context}
    else:
        kwargs = {'classes': classes, 'pretrained': True, 'ctx': context}
    net = get_model(model_name, **kwargs).features

    net.hybridize()
    net.forward(
        mx.nd.ones((1, 3, 32, 32), ctx=context[0]))
    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)
    # Note: Copy parameters from net into siamese. This will make training unconvergeble....
    # else:
    #     net_params = net.collect_params()
    #     siamesenet_params = siamesenet.collect_params()
    #     for p1, p2 in zip(net_params.values(), siamesenet_params.values()):
    #         p2.set_data(p1.data())

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = os.path.join(opt.save_dir, "params")
        log_dir = os.path.join(opt.save_dir, "logs")
    else:
        save_dir = 'params'
        log_dir = 'logs'
        save_period = 0
    makedirs(save_dir)
    makedirs(log_dir)

    def test(val_data, ctx, epoch):
        embedding = None
        labels = None
        images = None
        initialized = False

        for i, (data, label) in enumerate(val_data):
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            outputs = mx.nd.concat(*outputs, dim=0)
            label = mx.nd.concat(*label, dim=0)
            if initialized:
                embedding = mx.nd.concat(*(embedding, outputs), dim=0)
                labels = mx.nd.concat(*(labels, label), dim=0)
            else:
                embedding = outputs
                labels = label
                initialized = True

        with SummaryWriter(logdir=log_dir) as sw:
            sw.add_embedding(tag='{}_tripletnet_semihard_{}'.format(opt.dataset, epoch), embedding=embedding, labels=labels, images=images)

    def train(train_data, val_data, epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]

        # with SummaryWriter(logdir=log_dir, verbose=False) as sw:
        #     sw.add_graph(tripletnet)

        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})
        # Init contrastive loss
        loss_fn = TripletSemiHardLoss()

        global_step = 0

        for epoch in range(epochs):
            train_loss = 0
            num_batch = len(train_data)

            tbar = tqdm(train_data)

            for i, batch in enumerate(tbar):
                batch_loss = 0
                data = mx.gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0, even_split=False)
                label = mx.gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0, even_split=False)
                with ag.record():
                    losses = []
                    for x, y in zip(data, label):
                        embs = net(x)
                        losses.append(loss_fn(embs, y))
                for l in losses:
                    l.backward()
                    batch_loss += l.mean().asscalar()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in losses])
                global_step += batch_size

                with SummaryWriter(logdir=log_dir, verbose=False) as sw:
                    sw.add_scalar(tag="BatchLoss", value=batch_loss, global_step=global_step)

            train_loss /= batch_size * num_batch
            with SummaryWriter(logdir=log_dir, verbose=False) as sw:
                sw.add_scalar(tag="TrainLoss", value=train_loss, global_step=global_step)

            if epoch % save_period == 0:
                # Test on first device
                print("Test and visualize")
                test(val_data, ctx, epoch)
                net.export("{}/{}".format(save_dir, model_name), epoch=epoch)

    train(triplet_dataset_train_loader, dataset_test_loader, opt.num_epochs, context)


if __name__ == '__main__':
    main()
