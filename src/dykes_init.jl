	#using Pkg
	#Pkg.status()
	#Pkg.add("HDF5")
	#Pkg.add("CUDA")
	#Pkg.add("JupyterFormatter")
	#Pkg.add("Random")

	using CUDA
	using Printf
	using BenchmarkTools
	using HDF5
	using Random:Random

