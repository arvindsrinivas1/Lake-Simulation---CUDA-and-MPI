# Lake-Simulation---CUDA-and-MPI
Wrote a simulation of ripples on lake after pebbles are dropped by centralized finite difference using MPI and CUDA


This program models the surface of a lake, where some pebbles have been thrown onto the surface. Centralized Finite Difference is used to inform an area on how it is affected because of it's neighbours. We have tried both 4 stencil and 9 stencil approach, with the 9 stencil approach being better. 

The grid is decomposed on to GPU into 2D Blocks.

I have also implemented an MPI version of the program where the grid is decomposed based on the processor rank, each node communicates boundary information to the neighbour and CUDA kernel is run during a time-step.