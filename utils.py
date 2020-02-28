"""

"""

import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mxnet import gluon
from mxnet.gluon.data.vision import transforms



def genearte_pascal_colormap(size=256):
    colormap_cv = np.zeros((size, 3), dtype=int)
    ind = np.arange(size, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap_cv[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    colormap_plt = []
    for color in colormap_cv:
        color_pt = []
        # cv:BGR -> plt:RGB
        color_pt.append(color[2] / 255.0)
        color_pt.append(color[1] / 255.0)
        color_pt.append(color[0] / 255.0)
        colormap_plt.append(color_pt)
    return colormap_plt


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_features(features, labels, xlim=None, ylim=None):
    # Use pascal mask color scheme
    label_set = set(labels)
    colors = genearte_pascal_colormap((len(label_set)))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for idx, label in enumerate(label_set):
        inds = np.where(labels == label)[0]
        ax.scatter(features[inds, 0], features[inds, 1], alpha=0.5, color=colors[idx])
    if xlim:
        ax.xlim(xlim[0], xlim[1])
    if ylim:
        ax.ylim(ylim[0], ylim[1])
    ax.legend(label_set)

    # you can get a high-resolution image as numpy array!!
    return get_img_from_fig(fig)


def extract_features(dataloader, net, ctx):
    # Output vector size is 2
    features = np.zeros((len(dataloader._dataset), 2))
    labels = np.zeros(len(dataloader._dataset))
    k = 0
    for i, (data, label) in enumerate(dataloader):
        # Split into ctx list
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        outputs = [net.get_feature(X).asnumpy() for X in data]
        # concatenate
        outputs_np = np.concatenate(outputs, axis=0)
        features[k:k+len(label)] = outputs_np
        labels[k:k+len(label)] = label.asnumpy()
        k += len(label)

    return features, labels

def get_transform(jitter_param=0.4, pca_noise=0.2):
    # Init transformer
    # See https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/data/data_augmentation.html
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param,
                                     hue=jitter_param),
        transforms.RandomLighting(alpha=pca_noise),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    return transform_train, transform_test