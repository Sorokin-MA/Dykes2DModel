using CUDA
using Printf
using HDF5
using Random
using Distributed
using Parameters
using LazyGrids
using Plots
using Interpolations
using Distributions

data_folder::String = "..\\data_test\\"
