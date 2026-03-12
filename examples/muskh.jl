"""
file to test d2dm preassure model
"""

using Revise
using Dykes2DModel
using CUDA
using Plots
using Random

using Interpolations

#using CoordRefSystems

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return [X, Y]
end


function idc(ix, iy, nx)
    return ((iy) * nx + ix + 1)
end


function log_println(str, level = 1)
    println(("-> "^level) * str)
    #pritnln(str)
end

function set_init_sigma(S, g, rhp_r, depth::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + rhp_r * g * (depth - (depth) * (Float32(iy) / Float32(ny)))
    return
end

function set_init_sigma_cald_yy(S, g, rhp_r, dx, Lx::Float32, Ly::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    a = Lx / 3.0
    x::Float32 = ix * dx - Lx / 2
    y::Float32 = Ly - ((Ly) * (Float32(iy + 1) / Float32(ny)))
    p_0 = 5.0
    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + (-p_0 / pi * (atan((x + a) / y) - atan((x - a) / y)))
    return
end

function set_init_sigma_cald_xx(S, g, rhp_r, dx, Lx::Float32, Ly::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    a = Lx / 3.0
    x::Float32 = ix * dx - Lx / 2
    y::Float32 = Ly - ((Ly) * (Float32(iy + 1) / Float32(ny)))
    p_0 = 5.0
    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + (-p_0 / pi * ((atan((x + a) / y) - atan((x - a) / y)) + (y * (x + a) / (y^2 + (x + a)^2)) - (y * (x - a)) / (y^2 + (x - a)^2)))
    return
end


function set_init_sigma_cald_xy(S, g, rhp_r, dx, Lx::Float32, Ly::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    a = Lx / 3.0
    x::Float32 = ix * dx - Lx / 2
    y::Float32 = Ly - ((Ly) * (Float32(iy + 1) / Float32(ny)))
    p_0 = 5.0
    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + (p_0 / pi * log((y^2 + (x + a)^2) / (y^2 + (x - a)^2)))
    return
end

function d2dm_pres_test()
#Init
	log_println("Init")
    Random.seed!(1234) #Setup random
    Lx = 20000 #x Length of area
    Ly = 20000 #y Length of area

    nx = 20001÷5 #x grid resolution
    ny = 6001÷5  #y grid resolution

    X_left_lim, X_right_lim = 0, 20000
    Z_left_lim, Z_right_lim = 0, 6000
    z_limit = 5000 #Surface level (from bottom)

	#Setup main params
    XX = range(X_left_lim, X_right_lim, nx)
    ZZ = range(Z_left_lim, Z_right_lim, ny)
    dx = (X_right_lim - X_left_lim) / nx
    dz = (Z_right_lim - Z_left_lim) / ny

	log_println("length of XX: $(length(XX))", 2)
	log_println("length of ZZ: $(length(ZZ))", 2)

    dyke_param = DykeParam(x=2000, y=1000, a=500, b=20, phi=pi/2)  #Param of first dyke

    xpoints::Vector{Float64} = Vector{Float64}(undef, 1)
    ypoints::Vector{Float64} = Vector{Float64}(undef, 1)

    l_vecs = 0 # Eigenvectors
    next_point_x = 0 #x of next dyke
    next_point_y = 0 #y of next dyke

	#For simple analytical solution
    #@cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma( Sxy_gpu, 9.8, 0.02650, Float32(y_limit), nx, ny)
    #@cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma(Sxx_gpu, 9.8, 0.02650, Float32(Y_right_lim), nx, ny)
    #@cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma(Syy_gpu, 9.8, 0.02650, Float32(Y_right_lim), nx, ny)


	Sxx_file = h5open("Sxx.h5", "r") 
	Szz_file = h5open("Szz.h5", "r") 
	Sxz_file = h5open("Sxz.h5", "r") 
	val = 5

	#Sxx
	#Read dataset
	log_println("Processing Sxx", 2)
	Sxx = read(Sxx_file["/Sxx"])
	Sxx_x = read(Sxx_file["/x"])
	Sxx_z = read(Sxx_file["/z"])
    x_range = range(0, maximum(Sxx_x)-minimum(Sxx_x), length=length(Sxx_x))
    z_range = range(0, maximum(Sxx_z)-minimum(Sxx_z), length=length(Sxx_z))
	# Create interpolation object
	itp = interpolate(Sxx, BSpline(Cubic(Line(OnGrid()))))
    itp = Interpolations.scale(itp, z_range, x_range)
	Sxx_x_plot = x_range[1:val:end]
	Sxx_z_plot = z_range[1:val:end]
	Sxx_plot = [itp(z, x) for z in Sxx_z_plot, x in Sxx_x_plot]


    #X_left_lim, X_right_lim = minimum(Sxx_x), maximum(Sxx_x)
    #Z_left_lim, Z_right_lim = minimum(Sxx_z), maximum(Sxx_z)


	#Szz
	#Read dataset
	log_println("Processing Szz", 2)
	Szz = read(Szz_file["/Szz"])
	Szz_x = read(Szz_file["/x"])
	Szz_z = read(Szz_file["/z"])
#    x_range = range(minimum(Szz_x), maximum(Szz_x), length=length(Szz_x))
#    z_range = range(minimum(Szz_z), maximum(Szz_z), length=length(Szz_z))
	# Create interpolation object
	itp = interpolate(Szz, BSpline(Cubic(Line(OnGrid()))))
    itp = Interpolations.scale(itp, z_range, x_range)
	#val = 1
	Szz_x_plot = x_range[1:val:end]
	Szz_z_plot = z_range[1:val:end]
	Szz_plot = [itp(z, x) for z in Szz_z_plot, x in Szz_x_plot]


	#Sxz
	#Read dataset
	log_println("Processing Sxz", 2)
	Sxz = read(Sxz_file["/Sxz"])
	Sxz_x = read(Sxz_file["/x"])
	Sxz_z = read(Sxz_file["/z"])
#    x_range = 10000+range(minimum(Sxz_x), maximum(Sxz_x), length=length(Sxz_x))
#    z_range = 6000+range(minimum(Sxz_z), maximum(Sxz_z), length=length(Sxz_z))
	# Create interpolation object
	itp = interpolate(Sxz, BSpline(Cubic(Line(OnGrid()))))
    itp = Interpolations.scale(itp, z_range, x_range)
	#val = 1
	Sxz_x_plot = x_range[1:val:end]
	Sxz_z_plot = z_range[1:val:end]
	Sxz_plot = [itp(z, x) for z in Sxz_z_plot, x in Sxz_x_plot]


#=
	test_plot = Plots.heatmap(Sxx_x_plot, Sxx_z_plot, Sxx_plot, 
			title="Sxx", layout=(2, 2),
			xlabel="X", ylabel="Z")
	Plots.heatmap!(test_plot, Szz_x_plot, Szz_z_plot, Szz_plot, 
			title="Szz", subplot=2,
			xlabel="X", ylabel="Z")

	Plots.heatmap!(test_plot, Sxz_x_plot, Sxz_z_plot, Sxz_plot, 
			title="Sxz", subplot=3,
			xlabel="X", ylabel="Z")
    display(test_plot)
	return
=#

	Sxx_gpu = CUDA.zeros(Float64, length(Sxx_x_plot) * length(Sxx_z_plot))
    Szz_gpu = CUDA.zeros(Float64, length(Szz_x_plot) * length(Szz_z_plot))
    Sxz_gpu = CUDA.zeros(Float64, length(Sxz_x_plot) * length(Sxz_z_plot))

	#=
    copyto!(Sxx, Sxx_gpu)
    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma_cald_yy(Syy_gpu, 9.8, 0.0265, dx, Float32(X_right_lim), Float32(Y_right_lim), nx, ny)
    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma_cald_xx(Sxx_gpu, 9.8, 0.0265, dx, Float32(X_right_lim), Float32(Y_right_lim), nx, ny)
    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma_cald_xy(Sxy_gpu, 9.8, 0.0265, dx, Float32(X_right_lim), Float32(Y_right_lim), nx, ny)
	=#
	
	copyto!(Sxx_gpu, Sxx_plot)
	copyto!(Szz_gpu, Szz_plot)
	copyto!(Sxz_gpu, Sxz_plot)

    log_println("Sxx_gpu  length: $(length(Sxx_gpu))", 2)
    log_println("Sxx_plot length: $(length(Sxx_plot))", 2)

#Main loop
	log_println("Main loop", 1)
	#Grid params for gput kernel calulations
    blockSize = (16, 16)
    gridSize = (Int64(floor((nx + blockSize[1] - 1) ÷ blockSize[1])), Int64(floor((ny + blockSize[2] - 1) ÷ blockSize[2])))

    for i in 1:20
        @time begin
            @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] insert_dyke_gpu!(Sxx_gpu, Szz_gpu, Sxz_gpu,
                x_range, z_range,
                nx, ny,
                dyke_param.a, dyke_param.b, dyke_param.P_in, dyke_param.x, dyke_param.y, dyke_param.phi)

            synchronize()

            copyto!(vec(Sxx_plot), Sxx_gpu)
            copyto!(vec(Szz_plot), Szz_gpu)
            copyto!(vec(Sxz_plot), Sxz_gpu)

            next_point_x, next_point_y, xpoints, ypoints, l_vecs = calc_cent_of_next_dyke(dyke_param, Sxx_plot, Szz_plot, Sxz_plot, XX, ZZ, z_limit, Z_right_lim, X_right_lim)

			log_println("Dyke #$i inserted!", 2)
			log_println("Coordinates - ($next_point_x, $next_point_y) inserted!", 2)
            #println("$next_point_x, $next_point_y, $xpoints, $ypoints")
            _, phi_tmp = d2dm_cart_to_polar(l_vecs[3], l_vecs[4])
            dyke_param = DykeParam(x=next_point_x, y=next_point_y, phi=phi_tmp)
        end
    end

    copyto!(vec(Sxx_plot), Sxx_gpu)
    copyto!(vec(Szz_plot), Szz_gpu)
    copyto!(vec(Sxz_plot), Sxz_gpu)



#Visualise
	log_println("Visualising", 1)
	test_plot = Plots.heatmap(Sxx_x_plot, Sxx_z_plot, Sxx_plot, 
			title="Sxx", layout=(2, 2),
			xlabel="X", ylabel="Z")
	Plots.heatmap!(test_plot, Szz_x_plot, Szz_z_plot, Szz_plot, 
			title="Szz", subplot=2,
			xlabel="X", ylabel="Z")
	Plots.heatmap!(test_plot, Sxz_x_plot, Sxz_z_plot, Sxz_plot, 
			title="Sxz", subplot=3,
			xlabel="X", ylabel="Z")
    Plots.quiver!(test_plot, [next_point_x, next_point_x], [next_point_y, next_point_y], quiver=([l_vecs[1] l_vecs[2]], [l_vecs[3], l_vecs[4]]), subplot=4, xlimit=[X_left_lim, X_right_lim], ylimit=[Z_left_lim, Z_right_lim])
    display(test_plot)

#=
    P_plot = Plots.heatmap(x=Sxx_x_plot, y=Sxx_z_plot, Sxx_plot, layout=(2, 2), title="Sxx", xlimit=[X_left_lim, X_right_lim], ylimit=[Z_left_lim, Z_right_lim])
    Plots.heatmap!(P_plot, x=Szz_x_plot, y=Szz_z_plot, Szz_plot, subplot=2, title="Szz", xlimit=[X_left_lim, X_right_lim], ylimit=[Z_left_lim, Z_right_lim])
    Plots.heatmap!(P_plot, x=Sxz_x_plot, y=Sxz_z_plot, Sxz_plot, subplot=3, title="Sxz", xlimit=[X_left_lim, X_right_lim], ylimit=[Z_left_lim, Z_right_lim])
    Plots.scatter!(P_plot, xpoints, ypoints, subplot=4, markersize=1, xlimit=[X_left_lim, X_right_lim], ylimit=[Z_left_lim, Z_right_lim], title="last dyke")
    #Plots.quiver!(P_plot, [next_point_x, next_point_x], [next_point_y, next_point_y], quiver=([l_vecs[1] l_vecs[2]], [l_vecs[3], l_vecs[4]]), subplot=4, xlimit=[X_left_lim, X_right_lim], ylimit=[X_left_lim, X_right_lim])
    display(P_plot)
=#

    #Plots.savefig(P_plot, "d2dm_muskh.png")

#End
	log_println("Success!!!", 1)
end
