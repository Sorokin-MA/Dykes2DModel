include("dykes_init.jl")
#include("dykes_structs.jl")

function meshgrid(x, y)
	X = [i for i in x, j in 1:length(y)]
	Y = [j for i in 1:length(x), j in y]
	return X, Y
end

"""
inflatcate of matlab meshgrid function
"""
function meshgrid_2(xin, yin)
	nx = length(xin)
	ny = length(yin)
	xout = zeros(ny, nx)
	yout = zeros(ny, nx)
	for jx = 1:nx
		for ix = 1:ny
			xout[ix, jx] = xin[jx]
			yout[ix, jx] = yin[ix]
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

function dykes_rand()
	Random.seed!(1111)

	gpuid = 0 #gpu id
	tyear = 365 * 24 * 3600 #seconds in year

	Lx::Float64 = 20000 # x size of area, m
	Ly::Float64 = 20000 # y size of area, m %20000
	Lx_Ly = Lx / Ly
	narrow_fact = 0.5
	dyke_x_W = 10000 #m 
	dyke_x_Wn = dyke_x_W * narrow_fact #m 
	dyke_a_rng = Vector{Int32}
	dyke_a_rng = [100, 1500] #m
	dyke_b_rng = [10, 20] #m

	dyke_x_rng = [(Lx - dyke_x_W) / 2, (Lx + dyke_x_W) / 2]
	dyke_x_rng_n = [(Lx - dyke_x_Wn) / 2, (Lx + dyke_x_Wn) / 2]

	dyke_y_rng = [7000, 12000]#dykes y distribution
	dyke_t_rng = [0.95 * pi / 2, 1.05 * pi / 2]#dykes time distribution
	dyke_to_sill = 13000#boundary where dykes turn yourself to sill, m
	dz = 10000#z dimension? i guess, m

	Lam_r = 1.5#thermal conductivity of rock, W/m/K
	Lam_m = 1.2#thermal conductivity of magma, W/m/K
	rho = 2650#density, kg/m^3
	Cp = 1350#scpecifiv heat capacity, J/kg/K
	Lheat = 3.5e5#Latent heat of melting, J/kg
	T_top::Float64 = 100#temperature at depth 5 km, C
	dTdy = 20#how fast temperature decreasing with depth, K/km
	T_magma::Float64 = 1050#magma intrusion temperature, C
	T_ch = 700#?
	Qv = 0.0030 * 1.e9 / tyear#m^3/s
	dt::Float64 = 10 * tyear#time
	calc_years = 400e3
	tfin::Float64 = calc_years * tyear
	terupt::Float64 = calc_years * tyear

	#Qv = (0.00411 * 1.e9 / tyear)*(78000.0/(tfin/tyear))#m^3/s


	Ly_eruption::Float64 = 2000 # m
	lam_r_rhoCp::Float64 = Lam_r / (rho * Cp) # m^2/s
	dT = 500 # K
	E = 1.56e10 # Pa
	nu::Float64 = 0.3

	# scales
	tsc = Ly^2 / lam_r_rhoCp # s

	# nondimensional
	tsh::Float64 = 0.85
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
	critVol_hist = [270, 50, 0.5, 0.02, 0.64, 0.02, 0.02, 0.7, 0.02, 0.001, 0.001, 0.06, 0.05, 0.02, 0.07, 0.052, 0.854, 0.026, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029]

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
	nt::Int32 = tfin / dt
	nout::Int32 = round(nt / 12)
	nt_erupt = terupt / dt
	nerupt::Int32 = 1
	println(nt)
	println(dt)
	println(typeof(dt))
	println(dt * nt / tyear)
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
	indx = findall(x -> (x > dyke_x_rng[1]) & (x < dyke_x_rng[2]), xs)
	indy = findall(y -> (y > dyke_y_rng[1]) & (y < dyke_y_rng[2]), ys)

	#print T
	C = zeros(nx, ny)

	#heatmap(xs, ys, transpose(T))

	Q = 0
	dyke_a = Vector{Float64}(undef, 0)
	dyke_b = Vector{Float64}(undef, 0)
	dyke_x = Array{Float64}(undef, 0)
	dyke_y = Array{Float64}(undef, 0)
	dyke_t = Array{Float64}(undef, 0)
	dyke_v = []
	Vtot = q * nt_erupt * dt
	Q_tsh = 0.5 * Vtot

	while Q < Vtot
		#dyke_a = [dyke_a dyke_a_rng[1] + diff(dyke_a_rng)*rand];
		append!(dyke_a, dyke_a_rng[1] .+ diff(dyke_a_rng, dims=1) .* rand_limited_2(0.5, 0.1))
		#dyke_b = [dyke_b dyke_b_rng[1] + diff(dyke_b_rng)*rand];
		append!(dyke_b, dyke_b_rng[1] .+ diff(dyke_b_rng, dims=1) .* rand_limited_2(0.5, 0.1))
		if Q < Q_tsh
			#dyke_x = [dyke_x dyke_x_rng[1] + diff(dyke_x_rng)*rand];
			append!(dyke_x, dyke_x_rng[1] .+ diff(dyke_x_rng, dims=1) .* rand_limited_2(0.5, 0.1))
		else
			append!(dyke_x, dyke_x_rng_n[1] .+ diff(dyke_x_rng_n, dims=1) .* rand_limited_2(0.5, 0.1))
		end
		dyke_y = append!(dyke_y, dyke_y_rng[1] .+ diff(dyke_y_rng, dims=1) .* rand_limited_2(0.5, 0.1))
		dyke_t = append!(dyke_t, dyke_t_rng[1] .+ diff(dyke_t_rng, dims=1) .* rand_limited_2(0.5, 0.1))
		dyke_v = append!(dyke_v, pi * last(dyke_a) * last(dyke_b))
		Q = Q + last(dyke_v)
	end
	#println("dyke_y = $dyke_y")

	dyke_v = vcat(0, cumsum(dyke_v))
	sz_dyke_v = sizeof(dyke_v)
	println("dyke_v size = $sz_dyke_v")
	dv = last(dyke_v) / nt_erupt

	#ndykes = diff(floor(interp1(dyke_v,1:length(dyke_v),0:dv:last(dyke_v))), dim=2);
	ndykes = Vector{Int32}(undef, 1)
	ndykes = Int32.(diff(floor.(interp1(dyke_v, 1:length(dyke_v), 0:dv:last(dyke_v))), dims=1))

	println("Debug")


	#println(typeof(nt))
	@assert length(ndykes) == nt_erupt
	ndykes[(length(ndykes)+1):nt] .= 0
	@assert length(ndykes) == nt
	dyke_npartcl = zeros(Int32, sum(ndykes))
	dyke_nmarker = zeros(Int32, sum(ndykes))
	#px_dyke      = cell(sum(ndykes),1);
	#py_dyke      = cell(sum(ndykes),1);
	#mx_dyke      = cell(sum(ndykes),1);
	#my_dyke      = cell(sum(ndykes),1);

	px_dyke = Vector{Any}(undef, sum(ndykes))
	py_dyke = Vector{Any}(undef, sum(ndykes))
	mx_dyke = Vector{Any}(undef, sum(ndykes))
	my_dyke = Vector{Any}(undef, sum(ndykes))

	dyke_t_idxs = findall(x -> x >= dyke_to_sill, dyke_y)
	dyke_t[dyke_t_idxs] = dyke_t[dyke_t_idxs] .+ pi / 2 #reverse dykes to sills

	for idyke = 1:sum(ndykes)
		a = dyke_a[idyke]
		b = dyke_b[idyke]
		dykex0 = dyke_x[idyke]
		dykey0 = dyke_y[idyke]
		st = sin(dyke_t[idyke])
		ct = cos(dyke_t[idyke])
		#markers
		dykexs = LinRange(-a, a, Int32(round(2 * a / dr)))
		dykeys = LinRange(-b, b, Int32(round(2 * b / dr)))
		if isempty(dykexs)
			dykexs = 0
		end
		if isempty(dykeys)
			dykeys = 0
		end
		dykex, dykey = meshgrid(dykexs, dykeys)
		dykex = reshape(dykex, length(dykex), 1)
		dykey = reshape(dykey, length(dykey), 1)
		#    dykex          = dykex(:);
		#    dykey          = dykey(:);

		#println(dykex)
		outside = (dykex .^ 2 / a^2 + dykey .^ 2 / b^2) .> 1 + eps(Float64)
		#	println(outside)
		#	println("dykex size = " * string(length(dykex)))
		#	println("dykex size = " * string(size(dykex)))
		dykex = dykex[.!outside]
		#	println("dykex size = " * string(length(dykex)))
		#	println("dykex size = " * string(size(dykex)))
		dykey = dykey[.!outside]
		px_dyke[idyke] = dykex0 .+ dykex .* ct - dykey .* st
		py_dyke[idyke] = dykey0 .+ dykex .* st + dykey .* ct
		dyke_npartcl[idyke] = length(px_dyke[idyke])

		# markers
		if (a / dmr) > 1
			dykemxs = LinRange(-a, a, Int32(round(2 * a / dmr)))
		else
			dykemxs = LinRange(0, 0, 0)
		end

		if (b / dmr) > 1
			dykemys = LinRange(-b, b, Int32(round(2 * b / dmr)))
		else
			dykemys = LinRange(-b, b, 0)
		end

		if length(dykemxs) <= 1
			dykemxs = LinRange(0, 0, 1)
		end
		if length(dykemys) <= 1
			dykemys = LinRange(0, 0, 1)
			#dykemys = [0]
		end

		dykemx, dykemy = meshgrid_2(dykemxs, dykemys)
		dykemx = reshape(dykemx, length(dykemx), 1)
		dykemy = reshape(dykemy, length(dykemy), 1)
		#    dykemx          = dykemx(:);
		#    dykemy          = dykemy(:);
		#    outside         = (dykemx.^2/a^2 + dykemy.^2/b^2) > 1+eps(Float64);
		outside = (dykemx .^ 2 / a^2 + dykemy .^ 2 / b^2) .> 1 + eps(Float64)
		#    dykemx(outside) = [];
		#    dykemy(outside) = [];
		dykemx = dykemx[.!outside]
		dykemy = dykemy[.!outside]

		mx_dyke[idyke] = dykex0 .+ dykemx .* ct .- dykemy .* st
		my_dyke[idyke] = dykey0 .+ dykemx .* st .+ dykemy .* ct
		dyke_nmarker[idyke] = length(mx_dyke[idyke])
	end

	println("Debug")

	#px_dykes     = (px_dyke);
	#py_dykes     = (py_dyke);

	#px_dykes     = cell2mat(px_dyke);
	#py_dykes     = cell2mat(py_dyke);

	px_dykes_float = Vector{Float64}(undef, 1)
	py_dykes_float = Vector{Float64}(undef, 1)

	#=
	for idyke = 1:sum(ndykes)
		px_dykes_float = vcat(px_dykes_float, px_dykes[idyke]);
		py_dykes_float = vcat(py_dykes_float, py_dykes[idyke]);
	end
	=#


	#println("px_dyke")
	#println(size(px_dyke))


	px_dykes = vcat([px_dyke[i] for i in 1:size(px_dyke, 1)]...)
	py_dykes = vcat([py_dyke[i] for i in 1:size(py_dyke, 1)]...)

	#println("ndykes")
	#println(sum(ndykes))

	#println("px_dykes")
	#println(size(px_dykes))
	#println("py_dykes")
	#println(size(py_dykes))


	#println("mx")
	#println(size(mx))


	#println("my")
	#println(size(my))

	#println("mx_dyke")
	#println(size(mx_dyke))

	mx = vcat(mx, vcat([mx_dyke[i] for i in 1:size(mx_dyke, 1)]...))
	my = vcat(my, vcat([my_dyke[i] for i in 1:size(my_dyke, 1)]...))

	#println(mx_dyke)
	#=
	for idyke = 1:sum(ndykes)
			mx = vcat(mx, mx_dyke[idyke]);
			my = vcat(my, my_dyke[idyke]);
	end
	=#

	#println("px_dykes len")
	#println(size(px_dykes))
	#println("mx len")
	#println(size(mx))
	#println(typeof(mx))

	#println(size(mx))
	#println(length(mx))

	mT = T_top .+ dTdy / 1e3 .* (Ly .- my)

	mT[(mx.>dyke_x_rng[1]).&(mx.<dyke_x_rng[2]).&(my.>dyke_y_rng[1]).&(my.<dyke_y_rng[2])] .= T_ch
	partcl_edges = vcat([Int32(0)], accumulate(+, dyke_npartcl))
	marker_edges = vcat([Int32(0)], accumulate(+, dyke_nmarker))

	#println("npartcl")
	#println(last(partcl_edges))

	#println("typeof(critVol)")
	#println(typeof(critVol))

	#println("typeof(ndykes)")
	#println(typeof(ndykes))
	sim_dir = "..\\d2dm_data\\"

	#save data
	particles_file_name = sim_dir * "pa.bin"

	fid = open(particles_file_name, "w")
	println(typeof(Lx), typeof(Ly), typeof(lam_r_rhoCp), typeof(lam_m_rhoCp), typeof(L_Cp), typeof(T_top), typeof(T_bot), typeof(T_magma), typeof(tsh), typeof(gamma), typeof(Ly_eruption), typeof(nu), typeof(G), typeof(dt_diff), typeof(dx), typeof(dy), typeof(eiter), typeof(pic_amount))
	write(fid, Lx, Ly, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, T_magma, tsh, gamma, Ly_eruption, nu, G, dt_diff, dx, dy, eiter, pic_amount, tfin)
	println(typeof(pmlt), typeof(nx), typeof(ny), typeof(nl), typeof(nt), typeof(niter), typeof(nout), typeof(nsub), typeof(nerupt), typeof(npartcl), typeof(nmarker), typeof(Nsample))
	write(fid, pmlt, nx, ny, nl, nt, niter, nout, nsub, nerupt, npartcl, nmarker, Nsample)
	write(fid, critVol)
	write(fid, ndykes)
	write(fid, partcl_edges)
	write(fid, marker_edges)
	close(fid)

	dykes_file_name = sim_dir * "dykes.bin"
	fid = open(dykes_file_name, "w")
	write(fid, dyke_a, dyke_b, dyke_x, dyke_y, dyke_t)
	close(fid)

	fname = @sprintf "%sgrid.%s.h5" sim_dir "0"^ndigits
	fid = h5open(fname, "w")
	fid["T"] = T
	fid["C"] = C
	close(fid)

	fname = @sprintf "%sgrid.%s.h5" sim_dir "0"^(ndigits + 1)
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
	fid["px_dykes"] = px_dykes
	fid["py_dykes"] = py_dykes
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

function dykes_rand_param(init_vp)
	log_to_buffer("Generating data!\n")

	Random.seed!(init_vp.seed)

	tyear = 365 * 24 * 3600 #seconds in year

	Lx::Float64 = init_vp.Lx # x size of area, m
	Ly::Float64 = init_vp.Ly # y size of area, m %20000
	Lx_Ly = Lx / Ly
	narrow_fact = init_vp.narrow_fact
	dyke_x_W = init_vp.dyke_x_W #m
	dyke_x_Wn = dyke_x_W * narrow_fact #m 
	dyke_a_rng = Vector{Int32}
	dyke_a_rng = [100, 1500] #m
	dyke_b_rng = [10, 20] #m

	dyke_x_rng = [(Lx - dyke_x_W) / 2, (Lx + dyke_x_W) / 2]
	dyke_x_rng_n = [(Lx - dyke_x_Wn) / 2, (Lx + dyke_x_Wn) / 2]

	dyke_y_rng = [7000, 13000]					#dykes y distribution
	dyke_t_rng = [0.95 * pi / 2, 1.05 * pi / 2]	#dykes time distribution
	dyke_to_sill = init_vp.dyke_to_sill			#boundary where dykes turn yourself to sill, m
	dz = init_vp.Lz								#z dimension? i guess, m

	Lam_r = init_vp.Lam_r  #thermal conductivity of rock, W/m/K
	Lam_m = init_vp.Lam_m  #thermal conductivity of magma, W/m/K
	rho = init_vp.rho #density, kg/m^3
	Cp = init_vp.Cp #scpecifiv heat capacity, J/kg/K
	Lheat = init_vp.L_heat #Latent heat of melting, J/kg
	T_top::Float64 = init_vp.T_top #temperature at depth 5 km, C
	dTdy = init_vp.dTdy #how fast temperature decreasing with depth, K/km
	T_magma::Float64 = init_vp.T_magma#magma intrusion temperature, C
	T_ch = init_vp.T_ch#?
	Qv = init_vp.Qv / tyear#m^3/s
	dt::Float64 = init_vp.dt * tyear#time
	calc_years = init_vp.calc_years
	tfin::Float64 = calc_years * tyear
	terupt::Float64 = calc_years * tyear

	#Qv = (0.00411 * 1.e9 / tyear)*(78000.0/(tfin/tyear))#m^3/s

	println(init_vp.dyke_type)

	Ly_eruption::Float64 = init_vp.Ly_eruption # m
	lam_r_rhoCp::Float64 = Lam_r / (rho * Cp) # m^2/s
	dT = init_vp.dT # K
	E = init_vp.E # Pa
	nu = init_vp.nu

	# scales
	tsc = Ly^2 / lam_r_rhoCp # s

	# nondimensional
	tsh=init_vp.tsh
	lam_m_lam_r = Lam_m / Lam_r
	gamma = init_vp.gamma

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

	critVol = 10*ones(1, 1000)
	critVol_size = size(init_vp.critVol)
	if(critVol_size == 0)
		log_to_buffer("Error! No data about eruptions found!\n")
		return nothing
	end
	critVol_hist = init_vp.critVol

	println(size(init_vp.critVol))
	critVol_h = @view critVol[1:critVol_size[1]]

	println(size(init_vp.critVol))
	println(size(critVol_h))

	#copy!(critVol_h, critVol_hist)
	#critVol[1:sizeof(init_vp.critVol)] = critVol_hist
	copy!(critVol_h, critVol_hist)


	println(critVol)
	#critVol(1:24) = critVol_hist(1:end);
	critVol = 10^9 * critVol / dz / (1 - gamma)

	#numerics
	#steph = 10
	#ny::Int32 = Int32(floor(Ly / steph))
	#nx::Int32 = Int32(floor(Lx_Ly * ny))
	steph = init_vp.steph
	ny::Int32 = init_vp.ny
	nx::Int32 = init_vp.nx
	nl::Int32 = init_vp.nl
	nmy = init_vp.nmy
	nmx = floor(Lmx / Lmy * nmy)
	pmlt::Int32 = init_vp.pmlt
	niter::Int32 = nx
	eiter::Float64 = init_vp.eiter
	CFL = init_vp.CFL
	pic_amount::Float64 = init_vp.pic_amount
	nt::Int32 = tfin / dt
	nout::Int32 = round(nt / init_vp.nout)
	nt_erupt = terupt / dt
	nerupt::Int32 = 1
	println(nt)
	println(dt)
	println(typeof(dt))
	println(dt * nt / tyear)

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

	px, py = meshgrid(pxs, pys)
	px = reshape(px, length(px), 1)
	py = reshape(py, length(py), 1)

	mxs = Omx:dmx:Omx+Lmx
	mys = Omy:dmy:Omy+Lmy
	mx, my = meshgrid(mxs, mys)

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
	indx = findall(x -> (x > dyke_x_rng[1]) & (x < dyke_x_rng[2]), xs)
	indy = findall(y -> (y > dyke_y_rng[1]) & (y < dyke_y_rng[2]), ys)

	#print T
	C = zeros(nx, ny)

	Q = 0
	dyke_a = Vector{Float64}(undef, 0)
	dyke_b = Vector{Float64}(undef, 0)
	dyke_x = Array{Float64}(undef, 0)
	dyke_y = Array{Float64}(undef, 0)
	dyke_t = Array{Float64}(undef, 0)
	dyke_v = []
	Vtot = q * nt_erupt * dt
	Q_tsh = 0.5 * Vtot

	log_to_buffer("Generating dykes...\n")
	while Q < Vtot
		#dyke_a = [dyke_a dyke_a_rng[1] + diff(dyke_a_rng)*rand];
		append!(dyke_a, dyke_a_rng[1] .+ diff(dyke_a_rng, dims=1) .* rand_limited_2(init_vp.dyke_nu, init_vp.dyke_dev, init_vp.dyke_type))
		#dyke_b = [dyke_b dyke_b_rng[1] + diff(dyke_b_rng)*rand];
		append!(dyke_b, dyke_b_rng[1] .+ diff(dyke_b_rng, dims=1) .* rand_limited_2(init_vp.dyke_nu, init_vp.dyke_dev, init_vp.dyke_type))
		if Q < Q_tsh
			#dyke_x = [dyke_x dyke_x_rng[1] + diff(dyke_x_rng)*rand];
			append!(dyke_x, dyke_x_rng[1] .+ diff(dyke_x_rng, dims=1) .* rand_limited_2(init_vp.dyke_nu, init_vp.dyke_dev, init_vp.dyke_type))
		else
			append!(dyke_x, dyke_x_rng_n[1] .+ diff(dyke_x_rng_n, dims=1) .* rand_limited_2(init_vp.dyke_nu, init_vp.dyke_dev, init_vp.dyke_type))
		end
		dyke_y = append!(dyke_y, dyke_y_rng[1] .+ diff(dyke_y_rng, dims=1) .* rand_limited_2(init_vp.dyke_nu, init_vp.dyke_dev, init_vp.dyke_type))
		dyke_t = append!(dyke_t, dyke_t_rng[1] .+ diff(dyke_t_rng, dims=1) .* rand_limited_2(init_vp.dyke_nu, init_vp.dyke_dev, init_vp.dyke_type))
		dyke_v = append!(dyke_v, pi * last(dyke_a) * last(dyke_b))
		Q = Q + last(dyke_v)
	end

	dyke_v = vcat(0, cumsum(dyke_v))
	sz_dyke_v = sizeof(dyke_v)
	println("dyke_v size = $sz_dyke_v")
	log_to_buffer(@sprintf("Number of dykes - %d\n", sz_dyke_v))
	dv = last(dyke_v) / nt_erupt

	ndykes = Vector{Int32}(undef, 1)
	ndykes = Int32.(diff(floor.(interp1(dyke_v, 1:length(dyke_v), 0:dv:last(dyke_v))), dims=1))

	@assert length(ndykes) == nt_erupt
	ndykes[(length(ndykes)+1):nt] .= 0
	@assert length(ndykes) == nt
	dyke_npartcl = zeros(Int32, sum(ndykes))
	dyke_nmarker = zeros(Int32, sum(ndykes))

	px_dyke = Vector{Any}(undef, sum(ndykes))
	py_dyke = Vector{Any}(undef, sum(ndykes))
	mx_dyke = Vector{Any}(undef, sum(ndykes))
	my_dyke = Vector{Any}(undef, sum(ndykes))

	dyke_t_idxs = findall(x -> x >= dyke_to_sill, dyke_y)
	dyke_t[dyke_t_idxs] = dyke_t[dyke_t_idxs] .+ pi / 2 #reverse dykes to sills

	log_to_buffer("Generating particles and markers...\n")
	for idyke = 1:sum(ndykes)
		a = dyke_a[idyke]
		b = dyke_b[idyke]
		dykex0 = dyke_x[idyke]
		dykey0 = dyke_y[idyke]
		st = sin(dyke_t[idyke])
		ct = cos(dyke_t[idyke])
		#markers
		dykexs = LinRange(-a, a, Int32(round(2 * a / dr)))
		dykeys = LinRange(-b, b, Int32(round(2 * b / dr)))
		if isempty(dykexs)
			dykexs = 0
		end
		if isempty(dykeys)
			dykeys = 0
		end
		dykex, dykey = meshgrid(dykexs, dykeys)
		dykex = reshape(dykex, length(dykex), 1)
		dykey = reshape(dykey, length(dykey), 1)

		outside = (dykex .^ 2 / a^2 + dykey .^ 2 / b^2) .> 1 + eps(Float64)

		dykex = dykex[.!outside]

		dykey = dykey[.!outside]
		px_dyke[idyke] = dykex0 .+ dykex .* ct - dykey .* st
		py_dyke[idyke] = dykey0 .+ dykex .* st + dykey .* ct
		dyke_npartcl[idyke] = length(px_dyke[idyke])

		# markers
		if (a / dmr) > 1
			dykemxs = LinRange(-a, a, Int32(round(2 * a / dmr)))
		else
			dykemxs = LinRange(0, 0, 0)
		end

		if (b / dmr) > 1
			dykemys = LinRange(-b, b, Int32(round(2 * b / dmr)))
		else
			dykemys = LinRange(-b, b, 0)
		end

		if length(dykemxs) <= 1
			dykemxs = LinRange(0, 0, 1)
		end
		if length(dykemys) <= 1
			dykemys = LinRange(0, 0, 1)
		end

		dykemx, dykemy = meshgrid_2(dykemxs, dykemys)
		dykemx = reshape(dykemx, length(dykemx), 1)
		dykemy = reshape(dykemy, length(dykemy), 1)

		outside = (dykemx .^ 2 / a^2 + dykemy .^ 2 / b^2) .> 1 + eps(Float64)

		dykemx = dykemx[.!outside]
		dykemy = dykemy[.!outside]

		mx_dyke[idyke] = dykex0 .+ dykemx .* ct .- dykemy .* st
		my_dyke[idyke] = dykey0 .+ dykemx .* st .+ dykemy .* ct
		dyke_nmarker[idyke] = length(mx_dyke[idyke])
	end

	px_dykes_float = Vector{Float64}(undef, 1)
	py_dykes_float = Vector{Float64}(undef, 1)

	px_dykes = vcat([px_dyke[i] for i in 1:size(px_dyke, 1)]...)
	py_dykes = vcat([py_dyke[i] for i in 1:size(py_dyke, 1)]...)

	mx = vcat(mx, vcat([mx_dyke[i] for i in 1:size(mx_dyke, 1)]...))
	my = vcat(my, vcat([my_dyke[i] for i in 1:size(my_dyke, 1)]...))

	mT = T_top .+ dTdy / 1e3 .* (Ly .- my)

	mT[(mx.>dyke_x_rng[1]).&(mx.<dyke_x_rng[2]).&(my.>dyke_y_rng[1]).&(my.<dyke_y_rng[2])] .= T_ch
	partcl_edges = vcat([Int32(0)], accumulate(+, dyke_npartcl))
	marker_edges = vcat([Int32(0)], accumulate(+, dyke_nmarker))

	sim_dir = "..\\d2dm_data\\"

	if(isdir(sim_dir) == false)
		mkdir(sim_dir)
	end

	#save data
	particles_file_name = sim_dir * "pa.bin"

	log_to_buffer("Writing results to output files...\n")
	fid = open(particles_file_name, "w")
	println(typeof(Lx), typeof(Ly), typeof(lam_r_rhoCp), typeof(lam_m_rhoCp), typeof(L_Cp), typeof(T_top), typeof(T_bot), typeof(T_magma), typeof(tsh), typeof(gamma), typeof(Ly_eruption), typeof(nu), typeof(G), typeof(dt_diff), typeof(dx), typeof(dy), typeof(eiter), typeof(pic_amount))
	write(fid, Lx, Ly, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, T_magma, tsh, gamma, Ly_eruption, nu, G, dt_diff, dx, dy, eiter, pic_amount, tfin)
	println(typeof(pmlt), typeof(nx), typeof(ny), typeof(nl), typeof(nt), typeof(niter), typeof(nout), typeof(nsub), typeof(nerupt), typeof(npartcl), typeof(nmarker), typeof(Nsample))
	write(fid, pmlt, nx, ny, nl, nt, niter, nout, nsub, nerupt, npartcl, nmarker, Nsample)
	write(fid, critVol)
	write(fid, ndykes)
	write(fid, partcl_edges)
	write(fid, marker_edges)
	close(fid)

	dykes_file_name = sim_dir * "dykes.bin"
	fid = open(dykes_file_name, "w")
	write(fid, dyke_a, dyke_b, dyke_x, dyke_y, dyke_t)
	close(fid)

	fname = @sprintf "%sgrid.%s.h5" sim_dir "0"^ndigits
	fid = h5open(fname, "w")
	fid["T"] = T
	fid["C"] = C
	close(fid)

	fname = @sprintf "%sgrid.%s.h5" sim_dir "0"^(ndigits + 1)
	fid = h5open(fname, "w")
	fid["T"] = T
	fid["C"] = C
	close(fid)

	fname = sim_dir * "particles.h5"
	fid = h5open(fname, "w")
	px_dataset = create_dataset(fid, "px", datatype(px), dataspace(px), chunk=size(px), deflate=5)
	py_dataset = create_dataset(fid, "py", datatype(py), dataspace(py), chunk=size(py), deflate=5)

	write(px_dataset, px)
	write(py_dataset, py)
	fid["px_dykes"] = px_dykes
	fid["py_dykes"] = py_dykes
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

	log_to_buffer("Generationg succeded\n")
end
