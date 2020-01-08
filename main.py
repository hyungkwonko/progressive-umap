
import umap

if __name__ == "__main__":

  # TOY DATA # (1797, 64)
  from sklearn.datasets import load_digits
  digits = load_digits()
  y = umap.UMAP(n_neighbors=5).fit_transform(digits['data'])
  # print(y)
  # print(f"y.shape: {y.shape}")
