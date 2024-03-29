# Computational Optimization
**Diving into algorithms used for the solution of optimization problems.**

#### In this repo we'll demonstrate some commonly used algorithms for optimization problems. We'll take a look at Presolve Techniques, Scaling Techniques and an Exterior Point Algorithm. All of them will be tested on different matrices. In addition, we'll go through various instances of traveling salesman (TSP) and similar problems.

### Briefly, the repo contains:

  • **(1) [Compressed Sparse Row (CSR)](https://github.com/christakakis/computational_optimization/tree/main/(1)%20Compressed%20Sparse%20Row%20(CSR)).** Representation of a matrix A by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of rows, and column indices.
  
  • **(2) [Compressed Sparse Column (CSC)](https://github.com/christakakis/computational_optimization/tree/main/(2)%20Compressed%20Sparse%20Column%20(CSC)).** Representation of a matrix A by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of columns, and row indices.
  
  • **(3) [Equilibration (Scaling Technique)](https://github.com/christakakis/computational_optimization/tree/main/(3)%20Eliminate%20k-ton%20Equality%20Constraints).** Rows and columns of a matrix A are multiplied by positive scalars and these operations lead to non-zero numerical values of similar magnitude.
  
  • **(4) [Eliminate k-ton Equality Constraints (Presolve Method)](https://github.com/christakakis/computational_optimization/tree/main/(4)%20Equilibration%20Technique).** Identifying and eliminating singleton, doubleton, tripleton, and more general k-ton equality constraints in order to reduce the size of the problem and discover whether a LP is unbounded or infeasible.
  
  • **(5) [Exterior Point Simplex-type Algorithm](https://github.com/christakakis/computational_optimization/tree/main/(5)%20Exterior%20Point%20Siplex-type%20Algorithm).** An implementation of Exterior Point Algorithm.
  
  • **(6) [Parser for various TSP and more type of problems](https://github.com/christakakis/computational_optimization/tree/main/(6)%20Parser%20for%20TSP%20and%20more%20type%20of%20problems).** With the help of **tsplib95** a complete parser was made to read instances of type TSP, HCP, ATSP, SOP, CVRP. Also it supports Edge_Weight_Types of EXPLICIT, EUC_2D, EUC_3D, XRAY1, XRAY2, GEO, ATT, UPPER_ROW, LOWER_ROW and many more. Main goal of this parser is to return important information about a selected problem in order to apply heuristics and metaheuristics later. It is important to mention that this work was part of a group project and my part was about Hamiltonian Cycle Problems (HCP). Contributors are mentioned inside the files.
  
  • **(7) [TSP solver - Heuristic algorithm for optimal tour](https://github.com/christakakis/computational_optimization/tree/main/(7)%20TSP%20solver%20-%20Heuristic%20algorithm%20for%20finding%20optimal%20tour).** With the help of **elkai library** and **TSP parser from code (6)**, Lin-Kernighan-Helsgaun heuristic algorithm is applied to HCP, TSP, ATSP, SOP files to find optimal tour and plot them.
  
  • **(8) [CVRP solver - Finding routes and their weights](https://github.com/christakakis/computational_optimization/tree/main/(8)%20CVRP%20solver%20-%20Finding%20routes%20and%20their%20weights).** With the help of **VRPy python framework** and **TSP parser from code (6)**, best routes for CVRP files are found,  as well as their weights.
  
  • **(9) [Arithmetic Mean (Scaling Technique)](https://github.com/christakakis/computational_optimization/tree/main/(9)%20Arithmetic%20Mean).** This method aims to decrease the variance between the nonzero elements in the coefficient matrix A. Each
row is divided by the arithmetic mean of the absolute value of the elements in that row and each column is divided by the arithmetic mean of the absolute value of the elements in that column.
  
This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Computational Optimization course of my MSc program.
