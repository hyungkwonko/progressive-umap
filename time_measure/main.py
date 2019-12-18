import umap
import os
import pickle
import numpy as np
import timeit
import gzip
from gensim.models.keyedvectors import KeyedVectors
from matplotlib import pyplot as plt

from umap.utils import measure_time

# t-SNE
# https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
from sklearn.manifold import TSNE
# from sklearn.datasets import load_digits

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

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
    plt.title('Fashion MNIST Embedded')
    plt.savefig("./{}.png".format(filename))

def run_umap(x, y, item, n_neighbors_list, min_dist=0.3, metric="euclidean", verbose=True):
    for i in n_neighbors_list:
        print("UMAP NEIGHBOR NUMBER: ", i)
        x_umap = umap.UMAP(n_neighbors=i, min_dist=min_dist, metric=metric, verbose=verbose).fit_transform(x)
        filename = "umap_result" + str(i) + "neighbors"
        draw_plot(x_umap, y, item, filename)

def run_umap2(x, y, item, min_dist_list=[0.1,0.05,0.01], verbose=True):
    for i in min_dist_list:
        print("MIN DIST LIST: ", i)
        x_umap = umap.UMAP(min_dist=i, verbose=verbose).fit_transform(x)
        filename = "umap_result" + str(i) + "mindist"
        draw_plot(x_umap, y, item, filename)


if __name__ == "__main__":

    # # TOY DATA # (1797, 64)
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', verbose=True).fit_transform(digits.data)

    # FASHION MNIST (6-70000, 784), 26MB
    # https://github.com/zalandoresearch/fashion-mnist
    x, y = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')
    x = np.append(x, x_test, axis=0)
    y = np.append(y, y_test, axis=0)
    # x = pca(x, no_dims=300).real
    item = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # UMAP run
    run_umap(x=x, y=y, item=item, n_neighbors_list=[2,5,10,20,50])
    # run_umap2(x=x, y=y, item=item, min_dist_list=[0.1,0.05, 0.01])
    x_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='correlation', verbose=True).fit_transform(x)
    draw_plot(x_umap, y, item, "umap_result")
    # t-SNE run
    x_tse = run_tsne(x)
    draw_plot(x_tse, y, item, "tsne_result")

    # CIFAR 10 (60000, 3072), 163MB
    # http://www.cs.toronto.edu/~kriz/cifar.html
    x2, y2, x_test2, y_test2 = get_CIFAR10_data(cifar10_dir = './data/cifar-10/')
    x2 = np.append(x2, x_test2, axis=0)
    y2 = np.append(y2, y_test2, axis=0)
    item2 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # UMAP run
    x_umap2 = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', verbose=True).fit_transform(x2)
    draw_plot(x_umap2, y2, item2, "umap_result2")
    # t-SNE run
    x_tse2 = run_tsne(x2)
    draw_plot(x_tse2, y2, item2, "tsne_result2")

    # WORD VECTOR (0.6M-3M, 300), 3.35GB
    # https://www.kaggle.com/sandreds/googlenewsvectorsnegative300
    word_vectors = KeyedVectors.load_word2vec_format('./data/google/GoogleNews-vectors-negative300.bin', binary=True)
    x3 = word_vectors.vectors[:600000,] # wv.shape (3,000,000, 300) -> (600,000, 300)

    # UMAP run
    x_umap3 = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', verbose=True).fit_transform(x3)
    # t-SNE run
    x_tse3 = run_tsne(x3)

    # plotData = data[33]
    # plotData = plotData.reshape(28, 28)
    # plt.gray()
    # plt.imshow(plotData)
    # plt.show()
