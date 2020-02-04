from umap import umap_
import numpy as np
import timeit
from gensim.models.keyedvectors import KeyedVectors

from utils import pca, run_umap, run_umap2, run_tsne, draw_plot, load_merge_cifar, load_merge_mnist
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

if __name__ == "__main__":

    # # TOY DATA # (1797, 64)
    # from sklearn.datasets import load_digits
    # digits = load_digits()
    # umap_.UMAP(n_neighbors=5, min_dist=0.3, local_connectivity=1, metric='correlation', verbose=True).fit_transform(digits.data)

    # FASHION MNIST (6-70000, 784), 26MB
    # https://github.com/zalandoresearch/fashion-mnist
    x, y = load_merge_mnist()
    # x = pca(x, no_dims=300).real
    item = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # UMAP run
    # run_umap(x=x, y=y, item=item, n_neighbors_list=[5])
    # run_umap(x=x, y=y, item=item, n_neighbors_list=[2,5,10,20,50])
    # run_umap2(x=x, y=y, item=item, min_dist_list=[0.1,0.05, 0.01])
    x_umap = umap_.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', verbose=True).fit_transform(x)
    draw_plot(x_umap, y, item, "umap_result")
    # t-SNE run
    # x_tse = run_tsne(x)
    # draw_plot(x_tse, y, item, "tsne_result")

    # CIFAR 10 (60000, 3072), 163MB
    # http://www.cs.toronto.edu/~kriz/cifar.html
    # x2, y2 = load_merge_cifar()
    # item2 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # UMAP run
    # run_umap(x=x2, y=y2, item=item2, n_neighbors_list=[5,20,50,100,200])
    # x_umap2 = umap_.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', verbose=True).fit_transform(x2)
    # draw_plot(x_umap2, y2, item2, "umap_result2")
    # # t-SNE run
    # x_tse2 = run_tsne(x2)
    # draw_plot(x_tse2, y2, item2, "tsne_result2")

    # # WORD VECTOR (0.6M-3M, 300), 3.35GB
    # # https://www.kaggle.com/sandreds/googlenewsvectorsnegative300
    # word_vectors = KeyedVectors.load_word2vec_format('./data/google/GoogleNews-vectors-negative300.bin', binary=True)
    # x3 = word_vectors.vectors[:600000,] # wv.shape (3,000,000, 300) -> (600,000, 300)

    # # UMAP run
    # x_umap3 = umap_.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation', verbose=True).fit_transform(x3)
    # # t-SNE run
    # x_tse3 = run_tsne(x3)

    # plotData = data[33]
    # plotData = plotData.reshape(28, 28)
    # plt.gray()
    # plt.imshow(plotData)
    # plt.show()
