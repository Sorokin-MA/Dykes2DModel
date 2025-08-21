"""
file to test d2dm preassure model
"""

using Revise
using Dykes2DModel
using CUDA
using Plots
using Random

#using CoordRefSystems

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return [X, Y]
end


function idc(ix, iy, nx)
    return ((iy) * nx + ix + 1)
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
    #init phase


    Random.seed!(1234)

    #grid params
    Lx = 20000
    Ly = 20000

    nx = 2000
    ny = 2000

    tmp = 1024

    nx = tmp
    ny = tmp

    dx::Float64 = Lx / (nx - 1)
    dy::Float64 = Ly / (ny - 1)

    xs = 0:dx:Lx
    ys = 0:dy:Ly

    X_left_lim, X_right_lim = 0, 10
    Y_left_lim, Y_right_lim = 0, 10
    y_limit = 9

    XX = range(X_left_lim, X_right_lim, nx)
    YY = range(Y_left_lim, Y_right_lim, ny)
    dx = (X_right_lim - X_left_lim) / nx
    dy = (Y_right_lim - Y_left_lim) / ny
    X_rec, Y_rec = meshgrid(XX, YY)


    Sxx = zeros(size(X_rec))
    Syy = zeros(size(X_rec))
    Sxy = zeros(size(X_rec))

    println(size(XX))
    println(size(Sxx))

    dyke_param = DykeParam(x=5, y=3, a=4, b=0.5, phi=pi / 2)
    xpoints::Vector{Float64} = Vector{Float64}(undef, 1)
    ypoints::Vector{Float64} = Vector{Float64}(undef, 1)
    l_vecs = 0
    next_point_x = 0
    next_point_y = 0


    Sxx_cpu = Array{Float64}(undef, nx * ny)


    #TODO: change initial preassure rho_r * g * z

    Sxx_gpu = CUDA.zeros(Float64, nx * ny)
    Syy_gpu = CUDA.zeros(Float64, nx * ny)
    Sxy_gpu = CUDA.zeros(Float64, nx * ny)



    #Sxx_gpu[i + ny * (y-1)] = rhp_r	* g * y;


    # Sxx_gpu = CuArray{Float64}(undef, nx*ny)
    # Syy_gpu = CuArray{Float64}(undef, nx*ny)
    # Sxy_gpu = CuArray{Float64}(undef, nx*ny)

    println(size(X_rec))
    println(size(Sxx_gpu))

    #Sxx_gpu = CuArray([x::Float64 for x in Sxx])
    #Syy_gpu =CuArray([x::Float64 for x in Syy])
    #Sxy_gpu =CuArray([x::Float64 for x in Sxy])

    blockSize = (16, 16)
    gridSize = (Int64(floor((nx + blockSize[1] - 1) ÷ blockSize[1])), Int64(floor((ny + blockSize[2] - 1) ÷ blockSize[2])))

    #@cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma( Sxy_gpu, 9.8, 0.02650, Float32(y_limit), nx, ny)

    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma(Sxx_gpu, 9.8, 0.02650, Float32(Y_right_lim), nx, ny)
    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma(Syy_gpu, 9.8, 0.02650, Float32(Y_right_lim), nx, ny)


    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma_cald_yy(Syy_gpu, 9.8, 0.0265, dx, Float32(X_right_lim), Float32(Y_right_lim), nx, ny)
    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma_cald_xx(Sxx_gpu, 9.8, 0.0265, dx, Float32(X_right_lim), Float32(Y_right_lim), nx, ny)
    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] set_init_sigma_cald_xy(Sxy_gpu, 9.8, 0.0265, dx, Float32(X_right_lim), Float32(Y_right_lim), nx, ny)

    for i in 1:25

        @time begin
            @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] insert_dyke_gpu!(Sxx_gpu, Syy_gpu, Sxy_gpu,
                XX, YY,
                nx, ny,
                dyke_param.a, dyke_param.b, dyke_param.P_in, dyke_param.x, dyke_param.y, dyke_param.phi)

            synchronize()

            copyto!(Sxx, Sxx_gpu)
            copyto!(Syy, Syy_gpu)
            copyto!(Sxy, Sxy_gpu)

            next_point_x, next_point_y, xpoints, ypoints, l_vecs = calc_cent_of_next_dyke(dyke_param, Sxx, Syy, Sxy, XX, YY, y_limit, Y_right_lim, X_right_lim)

            println("Dyke #$i inserted!")
            _, phi_tmp = d2dm_cart_to_polar(l_vecs[3], l_vecs[4])
            dyke_param = DykeParam(x=next_point_x, y=next_point_y, phi=phi_tmp)
        end
    end

    copyto!(Sxx, Sxx_gpu)
    copyto!(Syy, Syy_gpu)
    copyto!(Sxy, Sxy_gpu)


    println(typeof(l_vecs))
    println(l_vecs)
    println(typeof(l_vecs[1]))

    P_plot = Plots.heatmap(x=XX', y=YY', Sxx', layout=(2, 2), title="Sxx")
    Plots.heatmap!(P_plot, x=XX', y=YY', Syy', subplot=2, title="Syy")
    Plots.heatmap!(P_plot, x=XX', y=YY', Sxy', subplot=3, title="Sxy")
    Plots.scatter!(P_plot, xpoints, ypoints, subplot=4, markersize=1, xlimit=[X_left_lim, X_right_lim], ylimit=[Y_left_lim, Y_right_lim], title="last dyke")
    #Plots.quiver!(P_plot, [next_point_x, next_point_x], [next_point_y, next_point_y], quiver=([l_vecs[1] l_vecs[2]], [l_vecs[3], l_vecs[4]]), subplot=4, xlimit=[X_left_lim, X_right_lim], ylimit=[X_left_lim, X_right_lim])

    display(P_plot)

    #Plots.savefig(P_plot, "d2dm_muskh.png")

    println("Success!!!")
end
