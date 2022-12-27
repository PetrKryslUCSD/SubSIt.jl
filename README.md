# SubSIt.jl

Subspace iteration algorithm for the generalized eigenvalue problem of free
vibration. 

Implementation of the Bathe Subspace Iteration algorithms from the textbook
Finite Element Procedures (1996),  Table 11.3.

A couple of twists have been added, which seem to improve convergence.
This package is sometimes slower than Arpack, but generally only 10-20%.
Other times it can be faster than  Arpack.

With contributions of Michael Stewart.
