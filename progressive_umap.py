
import umap
import pynene
from sklearn.datasets import load_digits


if __name__ == "__main__":

  digits = load_digits() # digits.data.shape = (1797, 64)
  umap.UMAP(n_neighbors=5, local_connectivity=1, metric='correlation', verbose=True).fit_transform(digits.data)

  # print(pynene.DTYPE)
  
