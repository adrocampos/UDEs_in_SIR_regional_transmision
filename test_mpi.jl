using MPI


MPI.Init()

comm = MPI.COMM_WORLD

MPI.Barrier(comm)


using OrdinaryDiffEq
using Optim
using CSV
using DataFrames


print("Hello world, I am rank $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))\n")