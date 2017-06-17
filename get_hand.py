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

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        #with self.init_scope():
        self.l1 = L.Linear(None, n_units)  # n_in -> n_units
        self.l2 = L.Linear(None, n_units)  # n_units -> n_units
        self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
#Loss関数
class LossFuncL(chainer.Chain):
    def __init__(self, predictor):
        super(LossFuncL, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        return loss
model = LossFuncL(MLP(args.unit, 1))
serializers.load_npz("mymodel.npz", model) # "mymodel.npz"の情報をmodelに読み込む

def hand(arr):
    return model.predictor(arr)

def get_hand(arr):
    return np.argmax(hand(arr))
if __name__ == '__main__':
    print("po")
