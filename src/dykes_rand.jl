include("dykes_init.jl")
include("dykes_structs.jl")

using LazyGrids
using Plots
using Interpolations

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end
"""
inflatcate of matlab meshgrid function
"""
function meshgrid_2(xin,yin)
nx=length(xin)
ny=length(yin)
xout=zeros(ny,nx)
yout=zeros(ny,nx)
for jx=1:nx
    for ix=1:ny
        xout[ix,jx]=xin[jx]
        yout[ix,jx]=yin[ix]
    end
end
return (x=xout, y=yout)
end


function interp1(xpt, ypt, x; method="linear", extrapvalue=nothing)

    if extrapvalue == nothing
        y = zeros(length(x))
        idx = trues(length(x))
    else
        y = extrapvalue * ones(x)
        idx = (x .>= xpt[1]) .& (x .<= xpt[end])
    end

    if method == "linear"
        intf = interpolate((xpt,), ypt, Gridded(Linear()))
        y[idx] = intf[x[idx]]

    elseif method == "cubic"
        itp = interpolate(ypt, BSpline(Cubic(Natural())), OnGrid())
        intf = scale(itp, xpt)
        y[idx] = [intf[xi] for xi in x[idx]]
    end

    return y
end

function dikes_rand()
    Random.seed!(1234)

    gpuid = 0#gpu id
    tyear = 365 * 24 * 3600#seconds in year

    Lx::Float64 = 20000 # x size of area, m
    Ly::Float64 = 20000 # y size of area, m %20000
    Lx_Ly = Lx / Ly
    narrow_fact = 0.6
    dike_x_W = 5000 #m 
    dike_x_Wn = dike_x_W * narrow_fact #m 
    dike_a_rng = Vector{Int32}
    dike_a_rng = [100, 1500] #m
    dike_b_rng = [10, 20] #m

    dike_x_rng = [(Lx - dike_x_W) / 2, (Lx + dike_x_W) / 2]
    dike_x_rng_n = [(Lx - dike_x_Wn) / 2, (Lx + dike_x_Wn) / 2]

    dike_y_rng = [1000, 22000]#dikes y distribution
    dike_t_rng = [0.95 * pi / 2, 1.05 * pi / 2]#dykes time distribution
    dike_to_sill = 21000#boundary where dykes turn yourself to sill, m
    dz = 5000#z dimension? i guess, m

    Lam_r = 1.5#thermal conductivity of rock, W/m/K
    Lam_m = 1.2#thermal conductivity of magma, W/m/K
    rho = 2650#density, kg/m^3
    Cp = 1350#scpecifiv heat capacity, J/kg/K
    Lheat = 3.5e5#Latent heat of melting, J/kg
    T_top::Float64 = 100#temperature at depth 5 km, C
    dTdy = 20#how fast temperature decreasing with depth, K/km
    T_magma::Float64 = 950#magma intrusion temperature, C
    T_ch = 700#?
    Qv = 0.038#m^3/s
    dt = 50 * tyear#time
    tfin::Int64 = 600e3 * tyear
    terupt::Int64 = 600e3 * tyear


    Ly_eruption::Float64 = 2000 # m
    lam_r_rhoCp::Float64 = Lam_r / (rho * Cp) # m^2/s
    dT = 500 # K
    E = 1.56e10 # Pa
    nu::Float64 = 0.3

    # scales
    tsc = Ly^2 / lam_r_rhoCp # s

    # nondimensional
    tsh::Float64 = 0.75
    lam_m_lam_r = Lam_m / Lam_r
    gamma::Float64 = 0.1

    Ste = dT / (Lheat / Cp) # Ste = dT/L_Cp

    # dimensionally dependent
    lam_m_rhoCp::Float64 = lam_r_rhoCp * lam_m_lam_r
    Omx = Lx / 2 - Lx / 3
    Lmx = 2 / 3 * Lx
    Omy = Ly / 2 - Ly / 3
    Lmy = 2 / 3 * Ly
    L_Cp::Float64 = dT / Ste

    q = Qv / dz
    G::Float64 = E / (2 * (1 + nu))

    alpha = 2 # parameter
    Nsample::Int32 = 1000 #size of a sample

    critVol = ones(1, 1000)
    critVol_hist = [265, 50, 0.5, 0.02, 0.64, 0.02, 0.02, 0.7, 0.02, 0.001, 0.001, 0.06, 0.05, 0.02, 0.07, 0.052, 0.854, 0.026, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029]

    critVol_h = @view critVol[1:24]
    copy!(critVol_h, critVol_hist)
    #critVol(1:24) = critVol_hist(1:end);
    critVol = 10^9 * critVol / dz / (1 - gamma)

    #numerics
    steph = 5
    ny::Int32 = Int32(floor(Ly / steph))
    nx::Int32 = Int32(floor(Lx_Ly * ny))
    nl::Int32 = 4
    nmy = 200
    nmx = floor(Lmx / Lmy * nmy)
    pmlt::Int32 = 2
    niter::Int32 = nx
    eiter::Float64 = 1e-12
    CFL = 0.23
    pic_amount::Float64 = 0.05
    nout::Int32 = 10000
    nt::Int32 = tfin / dt
    nt_erupt = terupt / dt
    nerupt::Int32 = 1

    #preprocessing
    dx::Float64 = Lx / (nx - 1)
    dy::Float64 = Ly / (ny - 1)
    dr = min(dx, dy) / pmlt
    dmx = Lmx / (nmx - 1)
    dmy = Lmy / (nmy - 1)
    dmr = min(dmx, dmy)
    xs = 0:dx:Lx
    ys = 0:dy:Ly
    x, y = meshgrid(xs, ys)
    nbd = floor(0.1 * (ny - 1))
    pxs = -nbd*dx-dx/pmlt/2:dx/pmlt:Lx+nbd*dx+dx/pmlt-dx/pmlt/2
    pys = -nbd*dy-dy/pmlt/2:dy/pmlt:Ly+nbd*dy+dy/pmlt-dy/pmlt/2
    #println("type of  pxs = ")
    #println(typeof(pxs))
    px, py = meshgrid(pxs, pys)
    px = reshape(px, length(px), 1)
    py = reshape(py, length(py), 1)
    #px          = px(:);
    #py          = py(:);
    mxs = Omx:dmx:Omx+Lmx
    mys = Omy:dmy:Omy+Lmy
    mx, my = meshgrid(mxs, mys)
    #mx          = mx(:);
    #my          = my(:);
    mx = reshape(mx, length(mx), 1)
    my = reshape(my, length(my), 1)

    dt_diff = CFL * min(dx, dy)^2 / lam_r_rhoCp
    nsub::Int32 = ceil(dt / dt_diff)
    dt_diff::Float64 = dt / nsub
    npartcl::Int32 = length(px)
    nmarker::Int32 = length(mx)
    T_bot::Float64 = T_top + dTdy * Ly / 1e3
    ndigits = Int32(floor(log10(nt))) + 1

    #init
    T = T_top .+ dTdy * (Ly .- y) / 1e3
    indx = findall(x -> (x > dike_x_rng[1]) & (x < dike_x_rng[2]), xs)
    indy = findall(y -> (y > dike_y_rng[1]) & (y < dike_y_rng[2]), ys)

    #print T
    C = zeros(nx, ny)

    heatmap(ys, xs, T)

    Q = 0
    dike_a = Vector{Float64}(undef, 0)
    dike_b = Vector{Float64}(undef, 0)
    dike_x = Array{Float64}(undef, 0)
    dike_y = Array{Float64}(undef, 0)
    dike_t = Array{Float64}(undef, 0)
    dike_v = []
    Vtot = q * nt_erupt * dt
    Q_tsh = 0.5 * Vtot

    while Q < Vtot
        #dike_a = [dike_a dike_a_rng[1] + diff(dike_a_rng)*rand];
        append!(dike_a, dike_a_rng[1] .+ diff(dike_a_rng, dims=1) .* rand(Float64, 1))
        #dike_b = [dike_b dike_b_rng[1] + diff(dike_b_rng)*rand];
        append!(dike_b, dike_b_rng[1] .+ diff(dike_b_rng, dims=1) .* rand(Float64, 1))
        if Q < Q_tsh
            #dike_x = [dike_x dike_x_rng[1] + diff(dike_x_rng)*rand];
            append!(dike_x, dike_x_rng[1] .+ diff(dike_x_rng, dims=1) .* rand(Float64, 1))
        else
            append!(dike_x, dike_x_rng_n[1] .+ diff(dike_x_rng, dims=1) .* rand(Float64, 1))
        end
        dike_y = append!(dike_y, dike_y_rng[1] .+ diff(dike_y_rng, dims=1) .* rand(Float64, 1))
        dike_t = append!(dike_t, dike_t_rng[1] .+ diff(dike_t_rng, dims=1) .* rand(Float64, 1))
        dike_v = append!(dike_v, pi * last(dike_a) * last(dike_b))
        Q = Q + last(dike_v)
    end
    #println("dike_y = $dike_y")

    dike_v = vcat(0, cumsum(dike_v))
    sz_dike_v = sizeof(dike_v)
    println("dike_v size = $sz_dike_v")
    dv = last(dike_v) / nt_erupt

    #ndikes = diff(floor(interp1(dike_v,1:length(dike_v),0:dv:last(dike_v))), dim=2);
    ndikes = Vector{Int32}(undef, 1)
    ndikes = Int32.(diff(floor.(interp1(dike_v, 1:length(dike_v), 0:dv:last(dike_v))), dims=1))

    #println(ndikes)


    #println(typeof(nt))
    @assert length(ndikes) == nt_erupt
    ndikes[(length(ndikes)+1):nt] .= 0
    @assert length(ndikes) == nt
    dike_npartcl = zeros(Int32, sum(ndikes))
    dike_nmarker = zeros(Int32, sum(ndikes))
    #px_dike      = cell(sum(ndikes),1);
    #py_dike      = cell(sum(ndikes),1);
    #mx_dike      = cell(sum(ndikes),1);
    #my_dike      = cell(sum(ndikes),1);

    px_dike = Vector{Any}(undef, sum(ndikes))
    py_dike = Vector{Any}(undef, sum(ndikes))
    mx_dike = Vector{Any}(undef, sum(ndikes))
    my_dike = Vector{Any}(undef, sum(ndikes))

    dike_t_idxs = findall(x -> x >= dike_to_sill, dike_y)
    dike_t[dike_t_idxs] = dike_t[dike_t_idxs] .+ pi / 2 #reverse dykes to sills

    for idike = 1:sum(ndikes)
        a = dike_a[idike]
        b = dike_b[idike]
        dikex0 = dike_x[idike]
        dikey0 = dike_y[idike]
        st = sin(dike_t[idike])
        ct = cos(dike_t[idike])
        #markers
        dikexs = LinRange(-a, a, Int32(round(2 * a / dr)))
        dikeys = LinRange(-b, b, Int32(round(2 * b / dr)))
        if isempty(dikexs)
            dikexs = 0
        end
        if isempty(dikeys)
            dikeys = 0
        end
        dikex, dikey = meshgrid(dikexs, dikeys)
        dikex = reshape(dikex, length(dikex), 1)
        dikey = reshape(dikey, length(dikey), 1)
        #    dikex          = dikex(:);
        #    dikey          = dikey(:);

        #println(dikex)
        outside = (dikex .^ 2 / a^2 + dikey .^ 2 / b^2) .> 1 + eps(Float64)
        #	println(outside)
        #	println("dikex size = " * string(length(dikex)))
        #	println("dikex size = " * string(size(dikex)))
        dikex = dikex[.!outside]
        #	println("dikex size = " * string(length(dikex)))
        #	println("dikex size = " * string(size(dikex)))
        dikey = dikey[.!outside]
        px_dike[idike] = dikex0 .+ dikex .* ct - dikey .* st
        py_dike[idike] = dikey0 .+ dikex .* st + dikey .* ct
        dike_npartcl[idike] = length(px_dike[idike])

        # markers
		if(a/dmr) > 1
			dikemxs = LinRange(-a, a, Int32(round(2 * a / dmr)))
		else
			dikemxs = LinRange(0,0,0)
		end

		if(b/dmr) > 1
        dikemys = LinRange(-b, b, Int32(round(2 * b / dmr)))
		else
        dikemys = LinRange(-b, b, 0)
		end

        if length(dikemxs) <= 1
            dikemxs = LinRange(0, 0, 1) 
        end
        if length(dikemys) <= 1
            dikemys = LinRange(0, 0, 1) 
            #dikemys = [0]
        end

        dikemx, dikemy = meshgrid_2(dikemxs, dikemys)
        dikemx = reshape(dikemx, length(dikemx), 1)
        dikemy = reshape(dikemy, length(dikemy), 1)
        #    dikemx          = dikemx(:);
        #    dikemy          = dikemy(:);
        #    outside         = (dikemx.^2/a^2 + dikemy.^2/b^2) > 1+eps(Float64);
        outside = (dikemx .^ 2 / a^2 + dikemy .^ 2 / b^2) .> 1 + eps(Float64)
        #    dikemx(outside) = [];
        #    dikemy(outside) = [];
        dikemx = dikemx[.!outside]
        dikemy = dikemy[.!outside]

        #FIXIT:
        mx_dike[idike] = dikex0 .+ dikemx .* ct .- dikemy .* st
        my_dike[idike] = dikey0 .+ dikemx .* st .+ dikemy .* ct
        dike_nmarker[idike] = length(mx_dike[idike])
    end

    #px_dikes     = (px_dike);
    #py_dikes     = (py_dike);

    #px_dikes     = cell2mat(px_dike);
    #py_dikes     = cell2mat(py_dike);

    px_dikes_float = Vector{Float64}(undef, 1)
    py_dikes_float = Vector{Float64}(undef, 1)

    #=
    for idike = 1:sum(ndikes)
    	px_dikes_float = vcat(px_dikes_float, px_dikes[idike]);
    	py_dikes_float = vcat(py_dikes_float, py_dikes[idike]);
    end
    =#


    #println("px_dike")
    #println(size(px_dike))


    px_dikes = vcat([px_dike[i] for i in 1:size(px_dike, 1)]...)
    py_dikes = vcat([py_dike[i] for i in 1:size(py_dike, 1)]...)

    #println("ndikes")
    #println(sum(ndikes))

    #println("px_dikes")
    #println(size(px_dikes))
    #println("py_dikes")
    #println(size(py_dikes))


    #println("mx")
    #println(size(mx))


    #println("my")
    #println(size(my))

    #println("mx_dike")
    #println(size(mx_dike))

    mx = vcat(mx, vcat([mx_dike[i] for i in 1:size(mx_dike, 1)]...))
    my = vcat(my, vcat([my_dike[i] for i in 1:size(my_dike, 1)]...))

    #println(mx_dike)
    #=
    for idike = 1:sum(ndikes)
    		mx = vcat(mx, mx_dike[idike]);
    		my = vcat(my, my_dike[idike]);
    end
    =#

    #println("px_dikes len")
    #println(size(px_dikes))
    #println("mx len")
    #println(size(mx))
    #println(typeof(mx))

    #println(size(mx))
    #println(length(mx))

    mT = T_top .+ dTdy / 1e3 .* (Ly .- my)

    mT[(mx.>dike_x_rng[1]).&(mx.<dike_x_rng[2]).&(my.>dike_y_rng[1]).&(my.<dike_y_rng[2])] .= T_ch
    partcl_edges = vcat([Int32(0)], accumulate(+, dike_npartcl))
    marker_edges = vcat([Int32(0)], accumulate(+, dike_nmarker))

    #println("npartcl")
    #println(last(partcl_edges))

    #println("typeof(critVol)")
    #println(typeof(critVol))

    #println("typeof(ndikes)")
    #println(typeof(ndikes))
    sim_dir = "..\\data_test\\"

    #save data
    particles_file_name = sim_dir * "pa.bin"

    fid = open(particles_file_name, "w")
    write(fid, Lx, Ly, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, T_magma, tsh, gamma, Ly_eruption, nu, G, dt_diff, dx, dy, eiter, pic_amount)
    write(fid, pmlt, nx, ny, nl, nt, niter, nout, nsub, nerupt, npartcl, nmarker, Nsample)
    write(fid, critVol)
    write(fid, ndikes)
    write(fid, partcl_edges)
    write(fid, marker_edges)
    close(fid)

    dikes_file_name = sim_dir * "dikes.bin"
    fid = open(dikes_file_name, "w")
    write(fid, dike_a, dike_b, dike_x, dike_y, dike_t)
    close(fid)

    fname = @sprintf "%sgrid.%s.h5" sim_dir "0"^ndigits
    fid = h5open(fname, "w")
    fid["T"] = T
    fid["C"] = C
    close(fid)

	fname = @sprintf "%sgrid.%s.h5" sim_dir "0"^(ndigits+1)
    fid = h5open(fname, "w")
    fid["T"] = T
    fid["C"] = C
    close(fid)

    #println(typeof(px))
    #println(size(px))
    fname = sim_dir * "particles.h5"
    fid = h5open(fname, "w")
    px_dataset = create_dataset(fid, "px", datatype(px), dataspace(px), chunk=size(px), deflate=5)
    py_dataset = create_dataset(fid, "py", datatype(py), dataspace(py), chunk=size(py), deflate=5)
    #fid["px"] = reshape(px,1,length(px))
    #fid["py"] = py
    write(px_dataset, px)
    write(py_dataset, py)
    fid["px_dikes"] = px_dikes
    fid["py_dikes"] = py_dikes
    close(fid)

    fname = sim_dir * "markers.h5"
    fid = h5open(fname, "w")
    create_group(fid, "0")
    fid_0 = fid["0"]
    fid_0["mx"] = mx
    fid_0["my"] = my
    fid_0["mT"] = mT
    close(fid)

    println("nx = $nx")
    println("ny = $ny")
    println("success!!!")
end
