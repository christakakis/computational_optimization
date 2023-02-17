# This work was part of a group project and the contributors are the following: 
# Konstantinos Pasvantis, Alexandra Gialama, 
# Manos Nikitas, Panagiotis Christakakis

!pip install tsplib95
!pip install vrpy

import tsplib95
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import random
from vrpy import VehicleRoutingProblem
from networkx import DiGraph, from_numpy_matrix, relabel_nodes, set_node_attributes
from numpy import array

def x_ray_1(a,b):
  x_ray_1 = tsplib95.distances.xray(a,b,sx=1.0, sy=1.0, sz=1.0)
  return x_ray_1
def x_ray_2(a,b):
  x_ray_2 = tsplib95.distances.xray(a,b,sx=1.25, sy=1.5, sz=1.15)
  return x_ray_2
def eudcl(a,b):
  eudcl = tsplib95.distances.euclidean(a,b)
  return eudcl
def att(a,b):
  att = tsplib95.distances.pseudo_euclidean(a,b)
  return att
def geo(a,b):
  geo = tsplib95.distances.geographical(a,b)
  return geo

#Explicit Functions

def upperRow(dim,Matrix_of_distances_1, problem):
  #Symmetrical TSP Explicit Upper row

  for i in range(0,dim-1):
        for j in range(0,len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][i])):
          Matrix_of_distances_1[i][j+i+1]= problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][i][j]
          Matrix_of_distances_1[j+i+1][i]=Matrix_of_distances_1[i][j+i+1]
  return(Matrix_of_distances_1)

def lowerRow(dim, Matrix_of_distances_1, problem):
  #Symmetrical TSP Explicit Lower row

  for i in range(0,dim-1):
        for j in range(0,len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][i])):
          Matrix_of_distances_1[i+1][j]= problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][i][j]
          Matrix_of_distances_1[j][i+1]=Matrix_of_distances_1[i+1][j]

  return(Matrix_of_distances_1)

def asymfullMatrix(dim, Matrix_of_distances_1, problem):
  #Assymetrical TSP Explicit (full matrix every time )

  float_formatter = "{:.0f}".format
  np.set_printoptions(formatter={'float_kind':float_formatter})
  all_numbers=[]
  for z in range(len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'])):
    for arithmoi_seiras in range(len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z])):
      all_numbers.append(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z][arithmoi_seiras])
  metritis_arithmou=0
  for i in range(dim):
    for j in range(dim):
      Matrix_of_distances_1[i][j]=all_numbers[metritis_arithmou]
      metritis_arithmou+=1

  return(Matrix_of_distances_1)

def sopfullMatrix(dim, Matrix_of_distances_1, problem):
  #SOP Explicit (full matrix every time )

  float_formatter = "{:.0f}".format
  np.set_printoptions(formatter={'float_kind':float_formatter})
  all_numbers=[]
  for z in range(1,len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'])):
    for arithmoi_seiras in range(len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z])):
      all_numbers.append(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z][arithmoi_seiras])
  metritis_arithmou=0
  for i in range(dim):
    for j in range(dim):
      Matrix_of_distances_1[i][j]=all_numbers[metritis_arithmou]
      metritis_arithmou+=1

  return(Matrix_of_distances_1)

#CVRP Explicit lower_col
def lowerCol(dim, Matrix_of_distances_1, problem):
  
  stoixeia= dim*(dim-1)/2
  all_numbers=[]
  for z in range(0,len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'])):
    for arithmos_seiras in range(len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z])):
      all_numbers.append(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z][arithmos_seiras])
  metritis_arithmou=0
  for i in range(dim-1,0,-1):
    list_0=np.zeros(dim-i)
    sthlh=all_numbers[:i]
    all_numbers=all_numbers[i:]
    list_0=np.append(list_0,sthlh)
    Matrix_of_distances_1[:,dim-i-1]=list_0
  
  for i in range(dim):
    for j in range(i,dim):
      Matrix_of_distances_1[i][j]=Matrix_of_distances_1[j,i]
  return(Matrix_of_distances_1)

#CVRP Explicit upper_col
def upperCol(dim, Matrix_of_distances_1, problem):
  
  stoixeia= dim*(dim-1)/2
  all_numbers=[]
  for z in range(0,len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'])):
    for arithmos_seiras in range(len(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z])):
      all_numbers.append(problem.as_keyword_dict()['EDGE_WEIGHT_SECTION'][z][arithmos_seiras])
  metritis_arithmou=0
  for i in range(1,dim):
    list_0=np.zeros(dim-i)
    sthlh=all_numbers[:i]
    all_numbers=all_numbers[i:]
    list_0=np.append(sthlh,list_0)
    Matrix_of_distances_1[:,i]=list_0
  for j in range(dim):
    for i in range(j,dim):
      Matrix_of_distances_1[i][j]=Matrix_of_distances_1[j,i]
  return(Matrix_of_distances_1)

"""**'TSP' function**"""

def tsp(path, prt=True, plot=True):
  """
  Main parser function that parses various TSP and more type
  of problems with the help of tsplib95.

  Works for the following:
    1. Hamiltonian cycle problem (HCP) instances,
    2. Asymmetric traveling salesman problem (ATSP) instances,
    3. Sequential ordering problem (SOP) instances,
    4. Capacitated vehicle problem (CVRP) instances,
    5. EDGE_WEIGHT_TYPE = 'EXPLICIT',
    6. EDGE_WEIGHT_TYPE = 'EUC_2D',
    7. EDGE_WEIGHT_TYPE = 'EUC_3D',
    8. EDGE_WEIGHT_TYPE = 'XRAY1',
    9. EDGE_WEIGHT_TYPE = 'XRAY2',
    10. EDGE_WEIGHT_TYPE = 'GEO',
    11.EDGE_WEIGHT_TYPE = 'ATT',
    12.EDGE_WEIGHT_FORMAT = 'UPPER_ROW',
    13.EDGE_WEIGHT_FORMAT = 'LOWER_ROW'
    14.EDGE_WEIGHT_FORMAT = 'UPPER_COL',
    15.EDGE_WEIGHT_FORMAT = 'LOWER_COL'
    16.EDGE_WEIGHT_FORMAT = 'FULL_MATRIX'
  """
  global dist, problem
  try:
    problem = tsplib95.load(path)
  except Exception as e:
    print(e)
    return None
  
  name = problem.name
  comment = problem.comment
  dim = problem.dimension
  Matrix_of_distances_1 = np.zeros((dim,dim))
  problem_type = problem.type
  try:
    dist = problem.as_keyword_dict()['EDGE_WEIGHT_TYPE']
  except:
    dist = ''
  problem_dict = {'Name': name, 'Type': problem_type, 
                  'Dimension': dim, 'Comment': comment,
                  'Distance': dist}

  distances = {'ATT': att, 'EUC_3D': eudcl, 'XRAY1': x_ray_1,
              'XRAY2': x_ray_2, 'GEO': geo,'EUC_2D': eudcl}

  if problem_type == "HCP":
    format = problem.edge_data_format
    Matrix_of_distances_1, node_from, node_to = hcp(dim, problem)

  elif dist in distances:
    for i in range(1,dim+1):
      for j in range(i+1,dim+1):
        Matrix_of_distances_1[i-1][j-1] = \
          distances[dist](problem.as_keyword_dict()['NODE_COORD_SECTION'][i], 
          problem.as_keyword_dict()['NODE_COORD_SECTION'][j])
        Matrix_of_distances_1[j-1][i-1] = Matrix_of_distances_1[i-1][j-1]

    if problem_type == "CVRP":
      problem_dict['Capacity'] = problem.as_keyword_dict()['CAPACITY']
      problem_dict["Depots"] = problem.as_keyword_dict()['DEPOT_SECTION']

  elif dist == 'EXPLICIT':
    # print("There is not distance between the points, we already have the distance matrix")
    # print("")
    try:
      format = problem.edge_weight_format
    except Exception as e:
      print(e)
    if format == 'UPPER_ROW':
      Matrix_of_distances_1 = upperRow(dim, Matrix_of_distances_1, problem)
    elif format == 'LOWER_ROW':
      Matrix_of_distances_1 = lowerRow(dim, Matrix_of_distances_1, problem)
    elif format == 'LOWER_COL':
      Matrix_of_distances_1 = lowerCol(dim, Matrix_of_distances_1, problem)
    elif format == 'UPPER_COL':
      Matrix_of_distances_1 = upperCol(dim, Matrix_of_distances_1, problem)
    elif format == 'FULL_MATRIX':
      if problem_type == 'ATSP':
        Matrix_of_distances_1 = asymfullMatrix(dim, Matrix_of_distances_1, problem)
      if problem_type == 'SOP':
        Matrix_of_distances_1 = sopfullMatrix(dim, Matrix_of_distances_1, problem)

  else:
    print('Unable to parse file')


  problem_dict['Weight Matrix'] = Matrix_of_distances_1

  if dist != 'EXPLICIT':
    problem_dict['Node Coordinates'] = problem.node_coords
        
  if plot:
    if problem_type == "HCP":
      plot_hcp(node_from, node_to)
    else:
      #plotTSP([path], problem_dict['Node Coordinates'])
      G = problem.get_graph()
      nx.draw_networkx(G, node_size=50, edgelist=[])
      plt.title("Network Graph of the Problem")
      plt.show()

  if prt:
    for key, val in problem_dict.items():
      if key not in ['Node Coordinates', 'Weight Matrix']:
        print(f'{key}: {val}')
      else:
        print(f'{key}:\n{val}')

  return problem_dict

def cvrp_solve(filename):

  problem_dict = tsp(filename, prt = False, plot = False)
  array = problem_dict['Weight Matrix']
  rows, cols = array.shape
  # Taking all the elements apart from the first column.
  weight_array = array[:,1:]
  # Inserting the first column into the last.
  weight_array = np.insert(weight_array, cols-1, array[:,0], axis = 1)
  # Inserting into the first column zeros.
  weight_array = np.insert(weight_array, 0, 0, axis = 1)
  # Inserting into the last row zeros.
  weight_array = np.insert(weight_array, rows, 0, axis = 0)
  # Define capacity, depots and demand variables.
  capacity = problem.as_keyword_dict()['CAPACITY']
  depots = problem.as_keyword_dict()['DEPOT_SECTION']
  demand = problem.as_keyword_dict()['DEMAND_SECTION']
  # Correcting indexes in demand section. 
  new_demand = {}
  for key in demand.keys():
    new_demand[key - 1] = demand[key]
  del new_demand[0]

  # The matrix is transformed into a DiGraph.
  A = np.array(weight_array, dtype = [("cost", float)])
  G = from_numpy_matrix(A, create_using = nx.DiGraph())
  # The demands are stored as node attributes.
  set_node_attributes(G, values = new_demand, name = "demand")
  # The depot is relabeled as Source and Sink.
  G = relabel_nodes(G, {0: "Source", rows: "Sink"})
  
  # Find and print the optimal(s) solution(s).
  prob = VehicleRoutingProblem(G, load_capacity = capacity)
  prob.solve()
  print("Best value for:", filename)
  print(prob.best_value, "\n")
  print("Best routes for:", filename)
  print(prob.best_routes, "\n")
  print("Weights for best possible routes:", filename)
  print(prob.best_routes_load)

# Example 1
cvrp_solve('eil7.vrp')

# Example 2
cvrp_solve('eil13.vrp')

# Example 3
cvrp_solve('eil22.vrp')