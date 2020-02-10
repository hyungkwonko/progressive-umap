## Results
We tested Progressive UMAP (PUMAP) on various benchmarks. Here I introduce some of the results. Detailed information on each dataset is explained [here](https://github.com/hyungkwonko/progressive-umap/blob/master/data/README.md).


### Fashion MNIST
About one third of the total embedding time was required for Progressive UMAP compared to original UMAP implementation. As illustrated, although the number of points embedded are fewer, the output visualization was almost the same.

|            UMAP             |            PUMAP          |
:----------------------------:|:--------------------------:
|     200 epochs (68.3s)       |     200 epochs (20.4s)   |
![Fashion MNIST umap](./images/fashion_umap.png)|![Fashion MNIST pumap](./images/fashion_pumap.png)


### MNIST

|            UMAP             |            PUMAP          |
:----------------------------:|:--------------------------:
|     200 epochs (65.9s)      |     200 epochs (19.8s)    |
![MNIST umap](./images/mnist_umap.png)|![MNIST pumap](./images/mnist_pumap.png)


### Kuzushiji MNIST
There is a problem in current PUMAP implementation. When appending new points, it locates them calculating the mean value of existing points for each class. As seen below, when one class is divided into big clusters more than one, newly inserted points harm the overall performance of embedding. We plan to solve it by locating them into one of the clusters' center, not some weird center of different clusters.

|            UMAP             |            PUMAP          |
:----------------------------:|:--------------------------:
|     200 epochs (68.4s)      |     200 epochs (47.9s)    |
![KMNIST umap](./images/kuzushiji_umap.png)|![KMNIST pumap](./images/kuzushiji_pumap.png)