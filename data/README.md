## Datasets

Progressive UMAP was tested utilizing various datasets. To download them, please use `download.sh` files in each folder. For example,

```sh
sh download.sh
```
or to remove downloaded files,
```sh
sh download.sh remove
```

Further information on each dataset can be found in the websites listed below.

<br>

## Summary

  datasets      | `row` | `column` | `class`
----------------|--------|--------|--------
  [toy data](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)     | 1,797  | 64 | 10
  [MNIST](http://yann.lecun.com/exdb/mnist/)     | 70,000  | 784 | 10
  [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)     | 70,000  | 784 | 10
  [Kuzushiji MNIST](https://github.com/rois-codh/kmnist)     | 70,000  | 784 | 10
  [Kuzushiji MNIST 49](https://github.com/rois-codh/kmnist)     | 270,912  | 784 | 49
  [Coil 20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)     | 1,440  | 16,384 | 20
  [Coil 100](http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php)     | 7,200  | 49,152 | 100
  [CIFAR 10](http://www.cs.toronto.edu/~kriz/cifar.html)     | 60,000  | 3072 | 10
  [Word Vector](https://www.kaggle.com/sandreds/googlenewsvectorsnegative300)     | 3M  | 300 | 
