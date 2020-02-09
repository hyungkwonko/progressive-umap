from umap import umap_panene
from utils import load_merge_mnist, load_merge_cifar, load_merge_kuzushiji, load_coil

if __name__ == "__main__":

  # x, y = load_merge_mnist(data="fashion")
  # umap_panene.UMAP(n_neighbors=5, random_state=5).fit_transform(X=x, y=None, label=y, dname='fashion', progressive=False)

  # x, y = load_merge_mnist(data="mnist")
  # umap_panene.UMAP(n_neighbors=5, random_state=3).fit_transform(X=x, y=None, label=y, dname='mnist', progressive=True)

  # x2, y2 = load_merge_cifar()
  # umap_panene.UMAP(n_neighbors=5).fit_transform(X=x2, y=None, label=y2, dname='cifar-10', progressive=True)

  # x3, y3 = load_merge_kuzushiji(data="kuzushiji")
  # umap_panene.UMAP(n_neighbors=5, random_state=12, first_ops=70000, ops=700).fit_transform(X=x3, y=None, label=y3, dname='kuzushiji', progressive=True)

  # x4, y4 = load_merge_kuzushiji(data="kuzushiji49")
  # umap_panene.UMAP(n_neighbors=5, random_state=30).fit_transform(X=x4, y=None, label=y4, dname='kuzushiji49', progressive=False)

  # x5, y5 = load_coil(data="coil20")
  # umap_panene.UMAP(n_neighbors=30, random_state=3, first_ops=2000, ops=20).fit_transform(X=x5, y=None, label=y5, dname='coil20', progressive=True)
  # umap_panene.UMAP(n_neighbors=30, random_state=3).fit_transform(X=x5, y=None, label=y5, dname='coil20', progressive=False)

  x6, y6 = load_coil(data="coil100")
  # umap_panene.UMAP(n_neighbors=30, random_state=3, first_ops=2000, ops=20).fit_transform(X=x5, y=None, label=y5, dname='coil20', progressive=True)
  umap_panene.UMAP(n_neighbors=30, random_state=3).fit_transform(X=x6, y=None, label=y6, dname='coil100', progressive=False)