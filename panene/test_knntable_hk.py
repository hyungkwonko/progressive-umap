
'''
pynene python library study code (KNNTable)

to deeply understand what each function means,
looking into C++ code is required
'''

import pynene
from pynene import KNNTable
from test_index_hk import random_vectors

# from sklearn.datasets import load_digits
import numpy as np

def test_table_size(n=30, d=10, k=5, ops=30):
  """
  SIZE
  ----------
  get the size of table

  Returns
  -------
  size: number of points inserted
  """
  array = random_vectors(n, d)
  neighbors = np.zeros((n, k), dtype=np.int64) # with np.in32, it is not optimized
  distances = np.zeros((n, k), dtype=np.float32)
  table = KNNTable(array, k, neighbors, distances)
  table.run(ops)
  print(table.size())

def test_table(n=30, d=10, k=5, ops=30):
  """
  KNNTable
  ----------
  BUILD KNNTable

  Parameters
  ----------
  array: number of points to add
  k: number of neighbors
  neighbors: index of kNN 2D array (n, k)
  distances: distance of kNN 2D array (n, k)
  treew: tree weight (addPointWeight, updateIndexWeight)
  tablew: table weight (treeWeight, tableWeight)
  checks: 

  Returns
  -------
  Part of below are referred from PANENE's progressive_knn_table.h 

  1) addNewPointOp: adds a new point P to both table and index.
  It requires:
  An insertion operation to the index (O(lg N))
  An insertion operation to the table (O(1))
  A knn search for P (O(klg(N)))
  Mark the neighbors of P as dirty and insert them to the queue

  2) updateIndexOp: updates a progressive k-d tree index.
  Basically, it calls the update function of the k-d tree and the update function
  creates a new k-d tree incrementally behind the scene

  3) updateTableOp: takes a dirty point from a queue and updates its neighbors.
  It requires:
  A knn search for P.
  Mark the neighbors of P as dirty and insert them to the queue

  ops = ops 1(addNewPointOp) + ops 2(updateIndexOp) + ops 3(updateTableOp)

  4) addPointResult: number of added points

  5) updateIndexResult: ?

  6) updateTableResult: number of points updated within the allocated operations.

  7) numPointsInserted: total number of points inserted

  8, 9, 10) elapsed time for each
  """
  # np.random.seed(10)
  array = random_vectors(n, d)
  neighbors = np.zeros((n, k), dtype=np.int64) # with np.in32, it is not optimized
  distances = np.zeros((n, k), dtype=np.float32)
  table = KNNTable(array, k, neighbors, distances)
    
  updates = table.run(ops)
  print(updates)
  print(neighbors)
  # print(distances)
  updates = table.run(ops)
  print(updates)
  print(neighbors)

def test_incremental_run(n=1000, d=10, k=5, ops=100):
  """
  test distances are sorted in increasing order
  """
  neighbors = np.zeros((n, k), dtype=np.int64)
  distances = np.zeros((n, k), dtype=np.float32)

  x = random_vectors(n)

  table = KNNTable(x, k, neighbors, distances)

  for i in range(n // ops):
    ur = table.run(ops)

    for nn in range(ur['numPointsInserted']):
      for kk in range(k - 1):
        if(distances[nn][kk] > distances[nn][kk+1]):
          raise ValueError("distances are not sorted")

def test_updates_after_all_points_added(n=100, d=10, k=5, w = (0.5, 0.5), ops=10):
  """
  test redistribution of ops work
  """
  x = random_vectors(n)
  neighbors = np.zeros((n, k), dtype=np.int64)
  distances = np.zeros((n, k), dtype=np.float32)
  table = KNNTable(x, k, neighbors, distances)

  for i in range(200):
    ur = table.run(ops)

    if ur['numPointsInserted'] >= n:
      print("current i: ", i)
      break

  for i in range(10):
    ur = table.run(ops)

    if(ur['addPointOps'] + ur['updateIndexOps'] + ur['updateTableOps'] != ops):
      raise ValueError("the sum of ops are not the same")

    if(ur['addPointOps'] + ur['updateIndexOps'] > w[0] * ops):
      raise ValueError("operations are not well residtributed")

if __name__ == "__main__":
  # test_table_size()
  test_table()
  # test_incremental_run()
  # test_updates_after_all_points_added()