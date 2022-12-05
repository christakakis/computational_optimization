# Computational Optimization
**Diving into algorithms used for the solution of optimization problems.**

#### In this repo we'll demonstrate some commonly used algorithms for optimization problems. We'll take a look at Presolve Techniques, Scaling Techniques and an Exterior Point Algorithm. All of them will be tested on different matrices.

### Briefly, the repo contains:

  • **Compressed Sparse Row (CSR).** Representation of a matrix A by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of rows, and column indices.
  
  • **Compressed Sparse Column (CSC).** Representation of a matrix A by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of columns, and row indices.
  
  • **Equilibration (Scaling Technique)** Rows and columns of a matrix A are multiplied by positive scalars and these operations lead to non-zero numerical values of similar magnitude.
  
  • **Eliminate k-ton Equality Constraints (Presolve Method)** Identifying and eliminating singleton, doubleton, tripleton, and more general k-ton equality constraints in order to reduce the size of the problem and discover whether a LP is unbounded or infeasible.
  
  • **Exterior Point Simplex-type Algorithm.** An implementation of Exterior Point Algorithm.

This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Computational Optimization course of my MSc program.
