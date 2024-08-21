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

data_folder::String = "..\\d2dm_data\\"

start_flag::Bool = false;
flag_break::Bool = false;
G_FLAG_INIT::Bool = true;

D2DM_STARTED::Bool = false;
D2DM_STOPED::Bool = true;

buf = "\n\nWelcome to Dykes2DModel!\n 1.Set parameters and upload history of eruptions \n 2. Generate dykes \n 3. Start calculations\n"
time_of_loop::Float64 = 0;
str_time_left = Time(0)

descr_Lx = "Lx\n\nThe size of the area along the x axis\n\nDimension: [m]"
descr_Ly = "Ly\n\nThe size of the area along the y axis\n\nDimension: [m]"
descr_Lz = "Lz\n\nThe size of the area along the z axis\n\nDimension: [m]"



global_EruptionVolumesVec = Vector{Float64}([1, 1, 1, 1, 1, 1, 265, 16, 50, 0.5, 0.02, 0.64, 0.02, 0.02, 0.7, 0.201, 0.06, 0.05, 0.02, 0.07, 0.930, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029])
global_EruptionTimesVec = Vector{Float64}([160.2, 109.3, 105.6, 102.5, 101.2, 91.8, 39.8, 29.3, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 10.6, 9.6, 9.3, 5.1, 4.7, 4.9, 4.5, 4.3, 4.2, 4.1, 3.9, 0.5])
