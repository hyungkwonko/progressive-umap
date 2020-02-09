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

---

## Reference

### TOY DATA, dimension = (1797, 64)
`from sklearn.datasets import load_digits`

### MNIST, dimension = (70000, 784), 11MB
`http://yann.lecun.com/exdb/mnist/`

### FASHION MNIST, dimension = (70000, 784), 26MB
`https://github.com/zalandoresearch/fashion-mnist`

### CIFAR 10, dimension = (60000, 3072), 163MB
`http://www.cs.toronto.edu/~kriz/cifar.html`

### WORD VECTOR, dimension = (3M, 300), 3.35GB
`https://www.kaggle.com/sandreds/googlenewsvectorsnegative300`

### Kuzushiji-MNIST
Kuzushiji-MNIST: (28x28 grayscale, 70,000 images, 10 classes)

Kuzushiji-49: (28x28 grayscale, 270,912 images, 49 classes)

`https://github.com/rois-codh/kmnist`


### COIL 20
`http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php`

### COIL 100
`http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php`