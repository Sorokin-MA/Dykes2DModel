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
using DelimitedFiles
using Base64
using Core: Typeof
using Dates

data_folder::String = "..\\data_test\\"

start_flag::Bool = false;
flag_break::Bool = false;
G_FLAG_INIT::Bool = true;
buf = "\n\nWelcome to Dykes2DModel!\n 1.Set parameters and upload history of eruptions \n 2. Generate dykes \n 3. Start calculations\n"
time_of_loop::Float64 = 0;
str_time_left = Time(0)

descr_Lx = "Lx\n\nThe size of the area along the x axis\n\nDimension: [m]"
descr_Ly = "Ly\n\nThe size of the area along the y axis\n\nDimension: [m]"
descr_Lz = "Lz\n\nThe size of the area along the z axis\n\nDimension: [m]"
