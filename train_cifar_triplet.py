import argparse, time, logging
import os
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from mxboard import SummaryWriter
from model_zoo.siamese import TripletNet
from dataloader import TripletDataset


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='training batch size per device (CPU/GPU).')
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
    parser.add_argument('--save-period', type=int, default=25,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='snapshots',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
    batch_size = opt.batch_size
    classes = 10

    # Init transformer
    # See https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/data/data_augmentation.html
    jitter_param = 0.4
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param,
                                     hue=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test_viz = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    dataset = opt.dataset
    if dataset == 'cifar10':
        dataset_train = gluon.data.vision.CIFAR10(train=True)
        dataset_test = gluon.data.vision.CIFAR10(train=False)
    elif dataset == 'cifar100':
        dataset_train = gluon.data.vision.CIFAR100(train=True, fine_label=True)
        dataset_test = gluon.data.vision.CIFAR100(train=False, fine_label=True)
    else:
        print("Dataset: {} is unknow".format(dataset))

    triplet_dataset_train = TripletDataset(dataset_train, transform=transform_train)
    triplet_dataset_train_loader = gluon.data.DataLoader(triplet_dataset_train, batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=opt.num_workers)

    dataset_test_loader = gluon.data.DataLoader(dataset_test.transform_first(transform_test), batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
    # TODO : Try normalizing but failed so we will loop through val set again to get data without normalization
    dataset_test_loader_2 = gluon.data.DataLoader(dataset_test.transform_first(transform_test_viz), batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)

    print("Number of train sample: {}".format(len(triplet_dataset_train)))
    print("Number of val sample: {}".format(len(dataset_test)))

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes, 'drop_rate': opt.drop_rate, 'pretrained': False, 'ctx': context}
    else:
        kwargs = {'classes': classes, 'pretrained': False, 'ctx': context}
    net = get_model(model_name, **kwargs)

    tripletnet = TripletNet(net.features)
    tripletnet.hybridize()
    tripletnet.initialize(mx.init.Xavier(), ctx=context)

    if opt.resume_from:
        tripletnet.load_parameters(opt.resume_from, ctx=context)
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

    def test(val_data, val_data_2, ctx, epoch):
        embedding = None
        labels = None
        images = None
        initialized = False

        for i, (data, label) in enumerate(val_data):
            if i >= 20:
                # only fetch the first 20 batches of images
                break
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            outputs = [tripletnet.get_feature(X) for X in data]
            outputs = mx.nd.concat(*outputs, dim=0)
            label = mx.nd.concat(*label, dim=0)
            if initialized:
                embedding = mx.nd.concat(*(embedding, outputs), dim=0)
                labels = mx.nd.concat(*(labels, label), dim=0)
            else:
                embedding = outputs
                labels = label
                initialized = True

        for i, (data, _) in enumerate(val_data_2):
            data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
            data = mx.nd.concat(*data, dim=0)
            if images is None:
                images = data
            else:
                images = mx.nd.concat(*(images, data), dim=0)

        with SummaryWriter(logdir=log_dir) as sw:
            sw.add_embedding(tag='{}_tripletnet_{}'.format(opt.dataset, epoch), embedding=embedding, labels=labels, images=images)

    def train(train_data, val_data, epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]

        tripletnet.forward(
            mx.nd.ones((1, 3, 32, 32), ctx=ctx[0]),
            mx.nd.ones((1, 3, 32, 32), ctx=ctx[0]),
            mx.nd.ones((1, 3, 32, 32), ctx=ctx[0]))
        with SummaryWriter(logdir=log_dir, verbose=False) as sw:
            sw.add_graph(tripletnet)

        trainer = gluon.Trainer(tripletnet.collect_params(), 'adam', {'learning_rate': 0.001})
        # Init contrastive loss
        loss_fn = gluon.loss.TripletLoss(margin=6)

        global_step = 0

        for epoch in range(epochs):
            train_loss = 0
            num_batch = len(train_data)

            tbar = tqdm(train_data)

            for i, batch in enumerate(tbar):
                batch_loss = 0

                img = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                img_pos = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                img_neg = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
                with ag.record():
                    output = [tripletnet(x1, x2, x3) for x1, x2, x3 in zip(img, img_pos, img_neg)]
                    loss = [loss_fn(x1, x2, x3) for x1, x2, x3 in output]
                for l in loss:
                    l.backward()
                    batch_loss += l.mean().asscalar()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])
                global_step += batch_size

                with SummaryWriter(logdir=log_dir, verbose=False) as sw:
                    sw.add_scalar(tag="BatchLoss", value=batch_loss, global_step=global_step)

            train_loss /= batch_size * num_batch
            with SummaryWriter(logdir=log_dir, verbose=False) as sw:
                sw.add_scalar(tag="TrainLoss", value=train_loss, global_step=global_step)

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                # Test on first device
                test(val_data, dataset_test_loader_2, ctx, epoch)
                tripletnet.save_parameters('{}/{}-{}.params'.format(save_dir, model_name, epoch))

        if save_period and save_dir:
            tripletnet.save_parameters('{}/{}-{}.params'.format(save_dir, model_name, epochs-1))

    train(triplet_dataset_train_loader, dataset_test_loader, opt.num_epochs, context)


if __name__ == '__main__':
    main()
