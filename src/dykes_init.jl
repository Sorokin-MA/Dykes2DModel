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
using Adapt

data_folder::String = "..\\d2dm_data\\"
path_to_snap::String = "c:\\"

if(isdir(data_folder) == false)
	mkdir(data_folder)
end

FLAG_make_snapshot::Bool = false;

start_flag::Bool = false;
flag_break::Bool = false;
G_FLAG_INIT::Bool = true;

D2DM_STARTED::Bool = false;
D2DM_STOPED::Bool = true;
D2DM_MARKERS::Bool = false;

buf = "\n\nWelcome to Dykes2DModel!\n 1.Set parameters and upload history of eruptions \n 2. Generate dykes \n 3. Start calculations\n"
time_of_loop::Float64 = 0;
str_time_spend::Float64 = 0;
str_time_left = Time(0)

descr_Lx = "Lx\n\nThe size of the area along the x axis\n\nDimension: [m]"
descr_Ly = "Ly\n\nThe size of the area along the y axis\n\nDimension: [m]"
descr_Lz = "Lz\n\nThe size of the area along the z axis\n\nDimension: [m]"



global_EruptionVolumesVec = Vector{Float64}([10, 10, 10, 10, 10, 10, 265, 16, 50, 0.5, 0.02, 0.64, 0.02, 0.02, 0.7, 0.201, 0.06, 0.05, 0.02, 0.07, 0.930, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029])
global_EruptionTimesVec = Vector{Float64}([160.2, 109.3, 105.6, 102.5, 101.2, 91.8, 39.8, 29.3, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 10.6, 9.6, 9.3, 5.1, 4.7, 4.9, 4.5, 4.3, 4.2, 4.1, 3.9, 0.5])


dykes_crystalinity = 1 .- Vector{Float64}([1, 1, 0.9904761904761905, 0.9439153277684773, 0.9280423118954614, 0.9121692960224456, 0.8772486611018104, 0.8306877983940973, 0.775661359514509, 0.7185185023716518, 0.6677248515780011, 0.6042327880859376, 0.5460317460317462, 0.4793650793650794, 0.41164017934647823, 0.35343908885168657, 0.2793650793650794, 0.2338624015687006, 0.19153432694692474, 0.13015873015873017, 0.09735446506076402, 0.08783065553695447, 0.08042321583581348, 0.06349206349206352, 0.04444444444444443, 0.022222222222222397, 0.009523809523809728, 0.008465576171874977,  0.007,  0.006,  0.005,  0.004,  0.003,  0.002, 0])
dykes_temp = Vector{Float64}([0, 703.9651009116576, 739.2545589231133, 747.5812835431677, 765.0277420751164, 786.8358424662811, 802.2997596178105, 812.609025618284, 820.9357502383384, 823.3148144154968, 828.0729427698136, 830.8485297437112, 831.8485297437112, 835.2101353012887, 837.5891994784471, 838.3996673898679, 839.5717771605048, 843.1403734262424, 851.0705752495575, 859.0007770728726, 878.4298132868789, 907.7715927046195, 924.4250419447282, 953.370334867368, 987.8667654361645, 1018.3980769424843, 1044.1712963961259, 1067.9619381677096, 1092.9421120278728, 1120.697836560295, 1139.3338634824613, 1157.176844811149, 1170.658184280621, 1185.7256149370496, 2000])

d2dm_crystal::Interpolations.MonotonicInterpolation = interpolate(dykes_temp, dykes_crystalinity, SteffenMonotonicInterpolation())



nodes = (dykes_temp,)
itp = interpolate(nodes, dykes_crystalinity, Gridded(Linear()))
cuitp = adapt(CuArray{eltype(dykes_temp)}, itp);

