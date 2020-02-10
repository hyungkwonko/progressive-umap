from umap import umap_
import os
import pickle
import numpy as np
import timeit
import gzip
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt
import imageio
import glob

from umap.utils import measure_time

# t-SNE
# https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
from sklearn.manifold import TSNE
from sklearn.utils import shuffle


def load_coil(data='coil20', seed=42):
    images, labels = [], []
    for path in glob.glob(os.path.join(os.getcwd(), 'data', data, 'files', '*.png')):
        labels.append(int(path.split("__")[0].split("/obj")[1]))
        images.append(imageio.imread(path))
    images, labels = np.array(images), np.array(labels)
    images, labels = images.reshape(images.shape[0], -1), labels.reshape(-1)
    return shuffle(images, labels, random_state=seed)

# Load Kuzushiji Japanese Handwritten dataset
def load_kuzushiji(path, dtype="kmnist", kind='train'):
    images_path = os.path.join(path, f'{dtype}-{kind}-imgs.npz')
    labels_path = os.path.join(path, f'{dtype}-{kind}-labels.npz')
    images = np.load(images_path)
    images = images.f.arr_0
    images = images.reshape(images.shape[0], -1)
    labels = np.load(labels_path)
    labels = labels.f.arr_0
    labels = labels.reshape(-1)
    return images, labels

def load_merge_kuzushiji(data='kuzushiji', seed=42):
    if data == 'kuzushiji':
        x, y = load_kuzushiji(os.path.join(os.getcwd(), 'data', data), kind='train')
        x_test, y_test = load_kuzushiji(os.path.join(os.getcwd(), 'data', data), kind='test')
    elif data == 'kuzushiji49':
        x, y = load_kuzushiji(os.path.join(os.getcwd(), 'data', data), dtype="k49", kind='train')
        x_test, y_test = load_kuzushiji(os.path.join(os.getcwd(), 'data', data), dtype="k49", kind='test')
    x = np.append(x, x_test, axis=0)
    y = np.append(y, y_test, axis=0)
    # return randomize data
    return shuffle(x, y, random_state=seed)

# FASHION MNIST (60000, 784), 26MB
def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def load_merge_mnist(data='mnist'):
    x, y = load_mnist(os.path.join(os.getcwd(), 'data', data), kind='train')
    x_test, y_test = load_mnist(os.path.join(os.getcwd(), 'data', data), kind='t10k')
    return np.append(x, x_test, axis=0), np.append(y, y_test, axis=0)

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

# CIFAR 10 (60000, 3072), 163MB
def load_pickle(f):
    return  pickle.load(f, encoding='latin1')

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(cifar10_dir):
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

def load_merge_cifar():
    x, y, x_test, y_test = get_CIFAR10_data(cifar10_dir = './data/cifar-10/')
    x_append = np.append(x, x_test, axis=0)
    y_append = np.append(y, y_test, axis=0)
    return x_append, y_append

@measure_time
def run_tsne(x, dim=2):
    tsne = TSNE(n_components=dim, random_state=0, verbose=1)
    y = tsne.fit_transform(x)
    return y

def draw_plot(x, y, item, filename):
    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*x.T, s=0.3, c=y, cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    cbar.set_ticks(np.arange(10))
    cbar.set_ticklabels(item)
    # plt.title('Fashion MNIST Embedded')
    plt.savefig(f"./{filename}.png")

def run_umap(x, y, item, n_neighbors_list, min_dist=0.05, verbose=True):
    for i in n_neighbors_list:
        print("UMAP NEIGHBOR NUMBER: ", i)
        x_umap = umap_.UMAP(n_neighbors=i, min_dist=min_dist, verbose=verbose).fit_transform(x)
        filename = "umap_result" + str(i) + "neighbors"
        draw_plot(x_umap, y, item, filename)

def run_umap2(x, y, item, min_dist_list=[0.1,0.05,0.01], verbose=True):
    for i in min_dist_list:
        print("MIN DIST LIST: ", i)
        x_umap = umap_.UMAP(min_dist=i, verbose=verbose).fit_transform(x)
        filename = "umap_result" + str(i) + "mindist"
        draw_plot(x_umap, y, item, filename)

