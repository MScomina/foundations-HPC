# Foundations of HPC exercises
Exercises for the exam "Foundations of HPC" - UniTS.

## Description
The exercises consist in implementing the requirements listed in [this repository](https://github.com/Foundations-of-HPC/High-Performance-Computing-2023/tree/main/ASSIGNMENTS) and [this repository](https://github.com/Foundations-of-HPC/Cloud-Basic-2023/blob/main/Assignments/Exercise.md) and writing a report describing the choices taken. <br>
To give a quick Tl;DR:
### Exercise 1: compare different OpenMPI algorithms for collective operations.
The idea is to compare the efficiency and efficacy of various broadcast and collective operations through the [OSU benchmark test](https://mvapich.cse.ohio-state.edu/benchmarks/).
More details found on the repository.
### Exercise 2: OpenMP and MPI programming
Exercise 2 splits into an MPI section (exercise 2a or 2b) and an OpenMP section (exercise 2c).<br>

#### MPI section: 
Either implement a broadcast or all-to-all algorithm (2a) in distributed memory or implement a parallel version of quick-sort (2b).
#### OpenMP section: 
Implementation of Mandelbrot set computation through OpenMP, and potential export onto a .pgm format image (2c).<br>

It is possible to ONLY implement a hybrid MPI+OpenMP version of 2c as the full requirement.
### Exercise 3: Cloud-Based File Storage System
The exercise consists in identifying, deploying, implementing a cloud-based file storage system with basic features expected of said cloud system (add/delete files, private folders/accounts) and address its scalability and deployability. More details on the repository. <br>