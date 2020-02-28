# coding=utf-8
import random
import mxnet as mx
import numpy as np
from gluoncv.data.base import dataset
from mxnet.gluon.data.sampler import Sampler

IS_SIMILAR = 0
IS_DISSIMILAR = 1
class PairDataset(dataset.Dataset):
    """
    Pair of Cifar dataset
    0: similar, 1 : dissimilar
    """
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self._transform = transform

        self._classes = set(self._dataset._label)
        self._label_to_indices = {label: np.where(self._dataset._label == label)[0] for label in self._classes}

    def __getitem__(self, index):
        # Generate positive + negative pair for training
        class_selected = self._dataset._label[index]
        img1 = self._dataset._data[index]

        should_get_same_class = random.randint(0, 1)
        # positive pair
        if should_get_same_class:
            class_pair = class_selected
            index_pair = index
            while index_pair == index:
                index_pair = random.choice(self._label_to_indices[class_selected])
        else:
            other_classes = list(self._classes).copy()
            other_classes.remove(class_selected)
            class_pair = random.choice(other_classes)
            index_pair = random.choice(self._label_to_indices[class_pair])
        img2 = self._dataset._data[index_pair]
        pair_label = mx.nd.array([int(class_selected != class_pair)])

        if self._transform is not None:
            return self._transform(img1), self._transform(img2), pair_label
        return img1, img2, pair_label

    def __len__(self):
        if self._train:
            return self._train_sample
        else:
            return np.shape(self._pairs)[0]


class TripletDataset(dataset.Dataset):
    """
    Naive triplet of dataset
    This will be used for naive triplet networking training
    For online mining approaches (batch all, batch hard) please use the TripletDatasetOnline class
    """
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self._transform = transform

        self._classes = set(self._dataset._label)
        self._label_to_indices = {label: np.where(self._dataset._label == label)[0] for label in self._classes}

    def __getitem__(self, index):
        # Generate positive + negative pair for training
        class_selected = self._dataset._label[index]
        img = self._dataset._data[index]

        # positive pair
        pos_index = index
        while pos_index == index:
            pos_index = random.choice(self._label_to_indices[class_selected])
        pos_img = self._dataset._data[pos_index]
        # negative pair
        other_classes = list(self._classes).copy()
        other_classes.remove(class_selected)
        neg_class = random.choice(other_classes)
        neg_index = random.choice(self._label_to_indices[neg_class])
        neg_img = self._dataset._data[neg_index]

        label = mx.nd.array([self._dataset._label[index]])

        if self._transform is not None:
            return self._transform(img), self._transform(pos_img), self._transform(neg_img), label
        return img, pos_img, neg_img, label

    def __len__(self):
        return len(self._dataset)


class BalanceBatchSampler(Sampler):

    def __init__(self, labels, n_classes=2, n_samples=2, last_batch='keep'):

        self._last_batch = last_batch
        self._prev = []

        self._labels = labels
        self._last_batch = last_batch
        self._labels_set = list(set(self._labels))
        self._label_to_indices = {label: np.where(self._labels == label)[0]
                                  for label in self._labels_set}
        for l in self._labels_set:
            np.random.shuffle(self._label_to_indices[l])
        self._used_label_indices_count = {label: 0 for label in self._labels_set}
        self._count = 0
        self._n_classes = n_classes
        self._n_samples = n_samples
        self._n_dataset = len(self._labels)
        self._batch_size = self._n_samples * self._n_classes

    def __iter__(self):
        self._count = 0
        while self._count + self._batch_size < self._n_dataset:
            classes = np.random.choice(self._labels_set, self._n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self._label_to_indices[class_][
                               self._used_label_indices_count[class_]:self._used_label_indices_count[
                                                                         class_] + self._n_samples])
                self._used_label_indices_count[class_] += self._n_samples
                if self._used_label_indices_count[class_] + self._n_samples > len(self._label_to_indices[class_]):
                    np.random.shuffle(self._label_to_indices[class_])
                    self._used_label_indices_count[class_] = 0
            yield indices
            self._count += self._n_classes * self._n_samples

    def __len__(self):
        return self._n_dataset // self._batch_size


if __name__ == '__main__':

    # Test creating cifar10pair
    from mxnet import gluon
    from mxnet.gluon.data.vision import transforms

    # Init transformer
    jitter_param = 0.4
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=[0.8, 1]),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param, hue=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    cifar10_dataset_train = gluon.data.vision.CIFAR10(train=True)
    cifar10_dataset_test = gluon.data.vision.CIFAR10(train=False)

    labels = cifar10_dataset_train._label
    sampler = BalanceBatchSampler(labels=labels, n_classes=8, n_samples=8, last_batch='discard')

    train_data = gluon.data.DataLoader(
        cifar10_dataset_train, batch_sampler=sampler,  num_workers=1)
    for idx, batch in enumerate(train_data):
        print("{}".format(len(batch)))

    # val_data = gluon.data.DataLoader(
    #     cifar10_pair_dataset_val,
    #     batch_size=512, shuffle=False, num_workers=1)
    # for idx, (img1, img2, label) in enumerate(val_data):
    #     print("{}".format(idx))
