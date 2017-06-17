#!/usr/bin/env python

from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import TupleDataset
from chainer import serializers
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=10000,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=200,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--frequency', '-f', type=int, default=-1,
                    help='Frequency of taking a snapshot')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=41,
                    help='Number of units')
args = parser.parse_args()

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_units),  # Third layer
            l4=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x, t=None, train=False):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = self.l4(h)

        if train:
            return F.mean_squared_error(h,t)
        else:
            return h

model = MLP(40,162,4)
serializers.load_npz("model2.npz", model) # "mymodel.npz"の情報をmodelに読み込む

def hand(arr):
    return model(arr)

def get_hand(arr,arr2):
    po=np.array([arr]).astype(np.float32)
    po2 = np.array([hand(po).data])[0][0]
    print(po2)
    return np.argmax(po2[:len(arr2)])+1
if __name__ == '__main__':
    print("po")
