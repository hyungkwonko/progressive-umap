
import umap

if __name__ == "__main__":

  # TOY DATA # (1797, 64)
  from sklearn.datasets import load_digits
  digits = load_digits()
  umap.UMAP(n_neighbors=5, min_dist=0.3, local_connectivity=1, metric='correlation', verbose=True).fit_transform(digits.data)

