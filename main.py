from umap import umap_panene
from utils import load_merge_mnist, load_merge_cifar

if __name__ == "__main__":

  x, y = load_merge_mnist()
  # item = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
  umap_panene.UMAP(n_neighbors=5, random_state=5).fit_transform(X=x, y=None, label=y, dname='zz', progressive=False)
  
  # x2, y2 = load_merge_cifar()
  # item2 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
  # umap_panene.UMAP(n_neighbors=5).fit_transform(X=x2, y=None, label=y2, item=item2, progressive=True)
