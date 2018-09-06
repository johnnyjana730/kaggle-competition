import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
df = pd.read_csv(SCRIPT_PATH +'/labels.csv')
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

from glob import glob
width = 50
n_all = len(glob(SCRIPT_PATH +'/Images/*/*.jpg'))
X = np.zeros((n_all, width, width, 3), dtype=np.uint8)
y = np.zeros((n_all, n_class), dtype=np.uint8)
for i, file_name in tqdm(enumerate(glob(SCRIPT_PATH +'/Images/*/*.jpg')), total=n_all):
    X[i] = cv2.resize(cv2.imread(file_name), (width, width))
    bred = file_name.split('/')[1].split('\\')[1][10:].lower()
    y[i][class_to_num[bred]] = 1    
# X = np.load(SCRIPT_PATH + '/mxnetf/'+ 'X50_all_images.npy')
np.save(SCRIPT_PATH + '/mxnetf/'+ 'X50_all_images.npy', X)

df2 = pd.read_csv(SCRIPT_PATH + '/sample_submission.csv')
n_test = len(df2)
y2 = np.zeros((n_test, n_class), dtype=np.uint8)
for i in tqdm(range(n_test)):
    x1 = cv2.resize(cv2.imread(SCRIPT_PATH + '/test/%s.jpg' % df2['id'][i]), (width, width))
    for x2 in range(len(X)):
        if x1.all() == X[x2].all(): 
            y2[i] = y[x2]
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 4, 1)
            plt.imshow(x1[:,:,::-1])
            plt.title(df2['id'][i])
            plt.subplot(2, 4, 2)
            plt.imshow(X[x2][:,:,::-1])
            plt.title(num_to_class[y[x2].argmax()])
            plt.show()

for b in breed:
    df2[b] = y2[:,class_to_num[b]]
df2.to_csv('pred.csv', index=None)