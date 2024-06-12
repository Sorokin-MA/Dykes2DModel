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
using DataStructures
using PlotlyJS
using DashBootstrapComponents

data_folder::String = "..\\data_test\\"

start_flag::Bool = false;
flag_break::Bool = false;
G_FLAG_INIT::Bool = true;
buf = "asdf\n"
