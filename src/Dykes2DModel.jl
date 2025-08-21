module Dykes2DModel


#Main funcs
export d2dm_init, d2dm_read_params, d2dm_check_melt_fracton,
    d2dm_update_T!, d2dm_make_snapshot, d2dm_update_T_NG!,
    d2dm_particles_injection,
    d2dm_g2p!, d2dm_p2g_interpolation,
    d2dm_inserting_dykes,
    d2dm_mf_rock, d2dm_dmf_rock,
    d2dm_mf_magma, d2dm_dmf_magma,
    d2dm_eruption_advection

#Structs
export InitVarParams, GridParams, VarParams

#Muskh
export DykeParam, insert_dyke_gpu!,
    d2dm_polar_to_cart, d2dm_cart_to_polar,
    calc_cent_of_next_dyke

#For generating dykes
export dykes_rand_param


#Init
#export global_EruptionVolumesVec


using Printf
using HDF5
using Random
using Distributed
#using Parameters
using LazyGrids
#using Plots
using Interpolations
using Distributions
#using DataStructures
using Adapt
using PlotlyJS
#using DashBootstrapComponents
#using DelimitedFiles
#using Base64
using Core: Typeof
using Dates
using CUDA
#using ForwardDiff
#using DataInterpolations
#using Dash

#for eigvecs
using LinearAlgebra

#Includes
include("init.jl")
include("structs.jl")
include("generate.jl")
include("all_funcs.jl")

#include("MechanicalKernel.jl")
include("muskh.jl")



#TODO
#NumericalKernel
#Solve heat equasion
#PhysicalKernel
#dmf, mf
#Structs


#TODO
#BaseFunction
#d2dm_generate_data(2 options - load to file, load to RAM)
#d2dm_load_data


#TODO
#Examples
#GUI


end
