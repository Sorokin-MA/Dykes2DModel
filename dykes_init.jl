    #using Pkg
    #Pkg.status()
    #Pkg.add("HDF5")
    #Pkg.add("CUDA")
    #Pkg.add("JupyterFormatter")
    using CUDA
    using Printf
    using BenchmarkTools
    using HDF5

