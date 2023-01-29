# This work was part of a group project and the contributors are the following: 
# Ioannis Kontizas, Ioannis Mastoras, Argyrios Koronaios, Georgia Kesesidou, 
# Antonios Kagias, Konstantinos Pasvantis, Alexandra Gialama, 
# Manos Nikitas, Christos Papadopoulos, Panagiotis Christakakis

!pip install tsplib95

import tsplib95
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import random

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

#HCP function

def hcp(dim, problem):
  Matrix_of_distances_1 = np.zeros((dim, dim))
  #Parse EDGE_DATA_SECTION to string, replace special characters and split it
  string = str(problem.as_keyword_dict()['EDGE_DATA_SECTION'])
  string = string.replace("{", "")
  string = string.replace("}", "")
  string = string.replace(":", "")
  string = string.replace("]", "")
  string = string.replace("[", ",")
  string = string.split(',')

  #Append each string as int number into a list
  nodes = []
  for item in string:
    for subitem in item.split():
        if(subitem.isdigit()):
            nodes.append(int(subitem))

  #Two separate lists: from and to nodes
  node_1 = []
  node_2 = []
  for i in range(0, (len(nodes)-1), 2):
    node_1.append(nodes[i])
  for j in range(1, len(nodes), 2):
    node_2.append(nodes[j])

  #Calculate Matrix of distances
  for i in range(len(node_1)):
    Matrix_of_distances_1[node_1[i]-1][node_2[i]-1] = 1

  return(Matrix_of_distances_1, node_1, node_2)

#Plotting Functions

#Plot TSP
def plotTSP(paths, points, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    """

    # Unpack the primary TSP path and transform it into a list of ordered 
    # coordinates

    print(paths)
    print(paths[0])

    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    
    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads
    a_scale = float(max(x))/float(150)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]), 
                    head_width = a_scale, color = 'r', 
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)

    #Set axis too slitghtly larger than the set of x and y
    plt.xlim(min(x)*0.95, max(x)*1.1)
    plt.ylim(min(y)*0.95, max(y)*1.1)
    plt.show()

#Plot HCP
def plot_hcp(node_1, node_2, threshold = 20):

  edges = list(zip(node_1, node_2))

  add_from = []
  for i in range(threshold):
    add_from.append(edges[i])

  G = nx.Graph()
  G.add_edges_from(add_from)
  pos = nx.spring_layout(G)
  nx.draw_networkx(G, edgelist=[]) 
  nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 40)
  nx.draw_networkx_labels(G, pos)
  plt.title("Network Graph of the Problem")
  plt.show()

# graph with depots 
  keyToFind = 'DEMAND_SECTION'

  try:
    if problem.has_key(keyToFind):
      #find depots
      k=[] #list with depots
      for i in range(1,dim):
        if problem.as_keyword_dict()["DEMAND_SECTION"][i]==0:
          k.append(i) #key depot
      color_map = []
      for node in G:
          if node in k:
              color_map.append('red')
          else: 
              color_map.append('blue')      
      nx.draw(G, node_color=color_map, with_labels=True,node_size=1000)
      plt.show()
      return
  except AttributeError:
    pass
  nx.draw_networkx(G, node_size=25, edgelist=[]) 
  plt.rcParams["figure.figsize"] = (16,14)
  plt.title("Nodes in space")
  plt.show()



"""'TSP function"""

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


# Function tsp calls for different type of problems to test results.

problem_dict = tsp('pr264.tsp')

problem_dict = tsp('eilB101.vrp')

problem_dict = tsp('alb5000.hcp')

problem_dict = tsp('ftv170.atsp')

problem_dict = tsp('ESC07.sop')

problem_dict = tsp('burma14.tsp')

problem_dict = tsp('eil7.vrp')

problem_dict = tsp('att48.vrp')