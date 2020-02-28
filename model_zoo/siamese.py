from __future__ import print_function

from mxnet import gluon
from mxnet.gluon import nn


class SiameseNet(gluon.HybridBlock):
    def __init__(self, embedding_net, **kwargs):
        super(SiameseNet, self).__init__(**kwargs)
        self.embedding_net = embedding_net

    def hybrid_forward(self, F, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_feature(self, x1):
        return self.embedding_net(x1)


class TripletNet(gluon.HybridBlock):

    def __init__(self, embedding_net, **kwargs):
        super(TripletNet, self).__init__(**kwargs)
        self.embedding_net = embedding_net

    def hybrid_forward(self, F, img, pos_img, neg_img):
        output1 = self.embedding_net(img)
        output2 = self.embedding_net(pos_img)
        output3 = self.embedding_net(neg_img)
        return output1, output2, output3

    def get_feature(self, img):
        return self.embedding_net(img)