from umap import umap_panene
from utils import load_merge_mnist, load_merge_cifar, load_merge_kuzushiji, load_coil

if __name__ == "__main__":

  x, y = load_merge_mnist(data="fashion")
  umap_panene.UMAP(n_neighbors=5, random_state=2, first_ops=15000, ops=1000).fit_transform(X=x, y=None, label=y, dname='fashion2', progressive=True)

  # umap_panene.UMAP(n_neighbors=5, random_state=2, first_ops=40000, ops=300).fit_transform(X=x, y=None, label=y, dname='fashion', progressive=True)
  # umap_panene.UMAP(n_neighbors=5, random_state=2).fit_transform(X=x, y=None, label=y, dname='fashion', progressive=False)

  # x, y = load_merge_mnist(data="mnist")
  # umap_panene.UMAP(n_neighbors=5, random_state=4, first_ops=40000, ops=300).fit_transform(X=x, y=None, label=y, dname='mnist', progressive=True)
  # umap_panene.UMAP(n_neighbors=5, random_state=4).fit_transform(X=x, y=None, label=y, dname='mnist', progressive=False)

  # x2, y2 = load_merge_cifar()
  # umap_panene.UMAP(n_neighbors=5).fit_transform(X=x2, y=None, label=y2, dname='cifar-10', progressive=True)

  # x3, y3 = load_merge_kuzushiji(data="kuzushiji", seed=12)
  # umap_panene.UMAP(n_neighbors=5, random_state=12, first_ops=100000, ops=300).fit_transform(X=x3, y=None, label=y3, dname='kuzushiji', progressive=True)
  # umap_panene.UMAP(n_neighbors=5, random_state=12).fit_transform(X=x3, y=None, label=y3, dname='kuzushiji', progressive=False)

  # x4, y4 = load_merge_kuzushiji(data="kuzushiji49", seed=30)
  # umap_panene.UMAP(n_neighbors=5, random_state=30, first_ops=400000, ops=100).fit_transform(X=x4, y=None, label=y4, dname='kuzushiji49', progressive=True)
  # umap_panene.UMAP(n_neighbors=5, random_state=30).fit_transform(X=x4, y=None, label=y4, dname='kuzushiji49', progressive=False)

  # x5, y5 = load_coil(data="coil20", seed=3)
  # umap_panene.UMAP(n_neighbors=50, random_state=3, first_ops=2500, ops=10).fit_transform(X=x5, y=None, label=y5, dname='coil20', progressive=True)
  # umap_panene.UMAP(n_neighbors=50, random_state=3).fit_transform(X=x5, y=None, label=y5, dname='coil20', progressive=False)

  x6, y6 = load_coil(data="coil100", seed=3)
  print(x6.shape)
  print(y6.shape)
  exit()
  umap_panene.UMAP(n_neighbors=10, random_state=3, first_ops=10000, ops=10).fit_transform(X=x6, y=None, label=y6, dname='coil100', progressive=True)
  # umap_panene.UMAP(n_neighbors=10, random_state=3).fit_transform(X=x6, y=None, label=y6, dname='coil100', progressive=False)