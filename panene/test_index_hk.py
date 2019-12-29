
'''
pynene python library study code (Index)

to deeply understand what each function means,
looking into C++ code is required
'''

import pynene
from pynene import Index
import numpy as np

def random_vectors(n=100, d=10, dtype=np.float32):
    return np.array(np.random.rand(n, d), dtype=dtype)

def test_index():
  """
  Index
  ----------
  BUILD KNN INDEX WE CAN WORK ON

  Parameters
  ----------
  array: number of points to add
  w: tree weight e.g.) (0.3, 0.7)
  reconstruction_weight: 
  trees:
  """
  x = random_vectors(n=100, d=10)
  index = Index(x)
  print(index.array.shape) # dim = (n, d) = (100, 10)

def test_add_points():
  """
  ADD_POINTS
  ----------
  WE MUST ADD POINTS BEFORE QUERYING THE INDEX

  Parameters
  ----------
  ops: number of points to add
  """
  x = random_vectors(n=30)
  index = Index(x)

  print(index.size()) # 0 since we did not add any points
  index.add_points(1) # add 1 point
  print(index.size()) # 1
  index.add_points(100000)
  print(index.size()) # 30 since we cannot add more than we have 

def test_knn_search():
  """
  KNN_SEARCH
  ----------
  GIVEN INDEX, RETURN INDEXES & DISTANCES in ASCENDING ORDER (including itself)

  Parameters
  ----------
  pid: index of target point
  k: number of points to find (WE MUST SET K LESS THAN OR EQUAL TO THE # OF POINTS)
  cores: number of cores to use
  checks:
  eps:
  sorted:

  Returns
  -------
  ids: ids of points found (numpy 2D array)
  dists: distances from target point (numpy 2D array)
  """
  x = random_vectors()
  index = Index(x)
  index.add_points(x.shape[0])

  # pick random integer
  pt = np.random.randint(x.shape[0]) # id. e.g.) 94
  print(x[[pt]]) # data. e.g.) [[0.64, ...]]

  idx, dist = index.knn_search(pt, 5, cores=1)
  print(idx) # if pt=10, array([[10, 80, 87,  5, 95]])
  print(dist) # array([[0, 0.76741797, 0.86952025, 0.90387696, 0.9157505 ]])

def test_knn_search_points():
  """
  KNN_SEARCH_POINTS
  ----------
  GIVEN DATA(ARRAY), RETURN INDEXES & DISTANCES in ASCENDING ORDER (including itself)

  Parameters
  ----------
  points: data(2d array) of target point (any 2d array can be possible) e.g.) [[0.33, 0.61, ...]]
  k: number of points to find (WE MUST SET K LESS THAN OR EQUAL TO THE # OF POINTS)
  cores: number of cores to use
  checks:
  eps:
  sorted:

  Returns
  -------
  ids: ids of points found (numpy 2D array)
  dists: distances from target point (numpy 2D array)
  """
  x = random_vectors(n=10, d=3)
  index = Index(x)
  index.add_points(x.shape[0])

  # pick random integer
  pt = np.random.randint(x.shape[0]) # id. e.g.) 94

  # TEST ON RANDOM DATA POINT
  pts = np.asarray(x[[pt]], dtype=np.float32)
  idx2, dist2 = index.knn_search_points(pts, 3, cores=1)
  print(idx2)
  print(dist2)

  # TEST ON WHOLE DATA SET (ARRAY)
  idx3, dist3 = index.knn_search_points(x, 5, cores=1)
  print(idx3)
  print(dist3)

def test_run():
  """
  RUN
  ----------
  INCREMENTALLY ADD POINTS TO THE TREE STRUCTURE

  Parameters
  ----------
  ops: number of points to add

  Returns
  -------
  numPointsInserted: total number of points inserted
  addPointOps: ?
  updateIndexOps: 
  addPointResult: ?
  updateIndexResult: 
  addPointElapsed: 
  updateIndexElapsed: 
  """
  x = random_vectors(n=30, d=3)
  index = Index(x, w=(0.5, 0.5))

  ops = 6

  for i in range(x.shape[0] // ops):
    ur = index.run(ops)
    print("===========")
    print("index.size(): ", index.size()) # index.size grows as we run iteratively
    print(ur)

def test_run2():
  '''
  Parameters
  ----------
  checks: number of nodes to check (?)
  '''
  n = 100
  k = 3
  ops = 10
  test_n = 1

  x = random_vectors(n)
  test_points = random_vectors(test_n)

  index = Index(x, w=(0.5, 0.5))
  
  for i in range(n // ops):
    ur = index.run(ops)
    print(ur)
    
    ids1, dists1 = index.knn_search_points(test_points, k, checks = 1)
    # ids2, dists2 = index.knn_search_points(test_points, k, checks = 50)
    ids3, dists3 = index.knn_search_points(test_points, k, checks = 100)            
    print("1: ", ids1)
    print("1: ", dists1)
    # print("2: ", ids2)
    print("3: ", ids3)
    print("3: ", dists3)
    print(index.size())

if __name__ == "__main__":

  # test_index()
  # test_add_points()
  # test_knn_search()
  # test_knn_search_points()
  # test_run()
  test_run2()