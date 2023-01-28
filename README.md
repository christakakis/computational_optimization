# Computational Optimization
**Diving into algorithms used for the solution of optimization problems.**

#### In this repo we'll demonstrate some commonly used algorithms for optimization problems. We'll take a look at Presolve Techniques, Scaling Techniques and an Exterior Point Algorithm. All of them will be tested on different matrices. In addition, we'll go through various instances of traveling salesman (TSP) and similar problems.

### Briefly, the repo contains:

  • **(1) Compressed Sparse Row (CSR).** Representation of a matrix A by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of rows, and column indices.
  
  • **(2) Compressed Sparse Column (CSC).** Representation of a matrix A by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of columns, and row indices.
  
  • **(3) Equilibration (Scaling Technique)** Rows and columns of a matrix A are multiplied by positive scalars and these operations lead to non-zero numerical values of similar magnitude.
  
  • **(4) Eliminate k-ton Equality Constraints (Presolve Method)** Identifying and eliminating singleton, doubleton, tripleton, and more general k-ton equality constraints in order to reduce the size of the problem and discover whether a LP is unbounded or infeasible.
  
  • **(5) Exterior Point Simplex-type Algorithm.** An implementation of Exterior Point Algorithm.
  
  • **(6) Parser for various TSP and more type of problems.** With the help of **tsplib95** a complete parser was made to read instances of type TSP, HCP, ATSP, SOP, CVRP. Also it supports Edge_Weight_Types of EXPLICIT, EUC_2D, EUC_3D, XRAY1, XRAY2, GEO, ATT, UPPER_ROW, LOWER_ROW and many more. Main goal of this parser is to return important information about a selected problem in order to apply heuristics and metaheuristics later. It is important to mention that this work was part of a group project and my part was about Hamiltonian Cycle Problems (HCP). Contributors are mentioned inside the files.
  
This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Computational Optimization course of my MSc program.
