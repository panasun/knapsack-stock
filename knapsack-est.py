from __future__ import print_function
from ortools.algorithms import pywrapknapsack_solver
from multiprocessing import Pool
import os
import numpy as np

hash_table = []

def knapsack(values, weights, W):
  solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    'Multi-dimensional solver')
  
  capacities = [W]
  solver.Init(values, weights, capacities)
  computed_value = solver.Solve()

  packed_items = []
  packed_weights = []
  total_weight = 0
  for i in range(len(values)):
    if solver.BestSolutionContains(i):
      packed_items.append(i)
      packed_weights.append(weights[0][i])
      total_weight += weights[0][i]
          
  packed_weights.sort(reverse=True)
  return (computed_value, packed_weights)

def randomKnapsack(values, W, N):
  np.random.shuffle(values)
  weights = [values]
  (computed_value, packed_weights) = knapsack(values, weights, W)
  hash_value = hash(tuple(packed_weights))
  print('%d' % (hash_value))
  if not hash_value in hash_table:
    hash_table.append(hash_value)
    return([computed_value, len(packed_weights), hash_value, packed_weights])
  else:
    return([None, None, None])

def montecarlo(values, W, N, processes):
  results = []
  pool = Pool(processes)
  for i in range(N):
    results.append(pool.apply_async(randomKnapsack, [values, W, N]))
  results = list(filter(lambda x: x[0] is not None, [result.get() for result in results]))
  results.sort(key=lambda x: x[1])

  print('Total Output: ', len(results))
  fo = open('output.txt', 'w')
  fo.write('Total Output: %d\n' % (len(results)))
  for r in results:
    fo.write('%d %d %d\n' % (r[0], r[1], r[2]))
    fo.write('%s\n' % ','.join(list(map(str, r[3]))))
    print(r[0:3])

def main():
  fp = open('input.txt')
  values_input = fp.read().split('\n')
  values = list(map(int, values_input))
  W = 301715126
  N = 1000
  processes = 4
  montecarlo(values, W, N, processes)

if __name__ == '__main__':
    main()
