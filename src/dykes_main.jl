include("dykes_init.jl")
include("dykes_funcs.jl")

#@device_code_warntype

#Pkg.generate("Dykes_2D")

function main()
	#Initialization of inner random
	Random.seed!(1234)

	#TODO:Настроить фильтр
	#TODO:Настроить девайс если не выбран
	
	print_gpu_properties()

	
	#Initialization of main variables
	Lx = 0.0				#X size of researched area (m)
	Ly = 0.0				#Y size of researched area (m)
	lam_r_rhoCp = 0.0		#Thermal conductivity of rock/(density*specific heat capacity)
	lam_m_rhoCp = 0.0		#Thermal conductivity of magma/(density*specific heat capacity)
	L_Cp = 0.0			#???#dT/Ste, Ste = dT/(Lheat/Cp); L_heat/Cp
	T_top = 0.0				#Temperature on the top of area (°C)
	T_bot = 0.0				#Temperature at the bottom of the area (°C)
	T_magma = 0.0			#Magma instrusion temperature(°C)
	tsh = 0.0			#???#nondimensonal 0.75
	gamma = 0.0			#???#nondimensional 0.1
	Ly_eruption = 0.0	#???#2000, m
	nu = 0.0				#Poisson ratio of rock
	G = 0.0				#???#E/(2*(1+nu));
	dt = 0.0			#???#time step
	dx::Float64 = 0.0		#X dimension step
	dy::Float64 = 0.0		#Y dimension step
	eiter = 0.0			#???#epsilon
	pic_amount = 0.0	#???#0.05

	pmlt = 0			#???#unused
	nx = 0					#Resolution for X dimension
	ny = 0					#Resolution for Y dimension
	nl = 0					#?
	nt = 0					#?
	niter = 0				#?
	nout = 0				#?
	nsub = 0				#?
	nerupt = 0				#how often check for eruptions
	npartcl = 0				#number of particles
	nmarker = 0				#number of markers
	nSample = 0				#size of a Sample, 1000

	filename = Array{Char,1}(undef, 1024)
	is_eruption = false


	dpa = Array{Float64,1}(undef, 18)	#array of double values from matlab script
	ipa = Array{Int32,1}(undef, 12)		#array of int values from matlab script


	io = open("data/pa.bin", "r")
	read!(io, dpa)
	read!(io, ipa)

	ipar = 1								#index to read parameters
	Lx, ipar = read_par(dpa, ipar)			#x length of area
	Ly, ipar = read_par(dpa, ipar)			#y length of area
	lam_r_rhoCp, ipar = read_par(dpa, ipar)	#a few coefficients multiplied
	lam_m_rhoCp, ipar = read_par(dpa, ipar)	#a few coefficients multipled
	L_Cp, ipar = read_par(dpa, ipar)		#
	T_top, ipar = read_par(dpa, ipar)
	T_bot, ipar = read_par(dpa, ipar)
	T_magma, ipar = read_par(dpa, ipar)
	tsh, ipar = read_par(dpa, ipar)
	gamma, ipar = read_par(dpa, ipar)
	Ly_eruption, ipar = read_par(dpa, ipar)
	nu, ipar = read_par(dpa, ipar)
	G, ipar = read_par(dpa, ipar)
	dt, ipar = read_par(dpa, ipar)
	dx, ipar = read_par(dpa, ipar)
	dy, ipar = read_par(dpa, ipar)
	eiter, ipar = read_par(dpa, ipar)
	pic_amount, ipar = read_par(dpa, ipar)

	ipar = 1

	pmlt, ipar = read_par(ipa, ipar)
	nx, ipar = read_par(ipa, ipar)
	ny, ipar = read_par(ipa, ipar)
	nl, ipar = read_par(ipa, ipar)
	nt, ipar = read_par(ipa, ipar)
	niter, ipar = read_par(ipa, ipar)
	nout, ipar = read_par(ipa, ipar)
	nsub, ipar = read_par(ipa, ipar)
	nerupt, ipar = read_par(ipa, ipar)
	npartcl, ipar = read_par(ipa, ipar)
	nmarker, ipar = read_par(ipa, ipar)
	nSample, ipar = read_par(ipa, ipar)

	critVol = Array{Float64,1}(undef, nSample) #???#Critical volume when eruption appears, predefined variable
	read!(io, critVol)

	#array 0 0 1 0 0 ... like, where 1 -instrusion
	ndikes = Array{Int32,1}(undef, nt)	#number of dykes intruded on n-th time step
	read!(io, ndikes)

	ndikes_all = 0

	#count all dykes
	for istep in 1:nt
		ndikes_all = ndikes_all + ndikes[istep]
	end

	println("ndikes_all")
	println(ndikes_all)

	#???
	particle_edges = Array{Int32,1}(undef, ndikes_all + 1)
	read!(io, particle_edges)

	#???
	marker_edges = Array{Int32,1}(undef, ndikes_all + 1)
	read!(io, marker_edges)

	close(io)

	#end of first part of reading

	cap_frac = 5.5  #???
	npartcl0 = npartcl #???
	max_npartcl = convert(Int64, npartcl * cap_frac) + particle_edges[ndikes_all+1] #???#count max particles

	println("max_npartcl")
	println(max_npartcl)

	nmarker0 = nmarker

	max_nmarker = nmarker + marker_edges[ndikes_all+1]


	#blockSize(16, 32);
	#gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);


	blockSize = (16,32)
	gridSize = (Int64(floor((nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((ny + blockSize[2] - 1) / blockSize[2])))


	T = CuArray{Float64,1}(undef, nx * ny)
	T_old = CuArray{Float64,1}(undef, nx * ny)
	C = CuArray{Float64,1}(undef, nx * ny)
	wts = CuArray{Float64,1}(undef, nx * ny)
	pcnt = CuArray{Int32,1}(undef, nx * ny)

	a = CuArray{Float64}(undef, (1, 2))

	px = CuArray{Float64}(undef, max_npartcl)	#x coordinate of particle
	py = CuArray{Float64}(undef, max_npartcl)	#y coordinate of particle
	pT = CuArray{Float64}(undef, max_npartcl)	#Temperature of particle
	pPh = CuArray{Int8}(undef, max_npartcl)		#???#Ph?

	np_dikes = particle_edges[ndikes_all+1]		#number of particles in each dike during intrusion

	px_dikes = CuArray{Float64,1}(undef, np_dikes)	#x of dykes particles
	py_dikes = CuArray{Float64,1}(undef, np_dikes)	#y of dykes particles
	

	mx = CuArray{Float64,1}(undef, max_nmarker)		#x of marker
	my = CuArray{Float64,1}(undef, max_nmarker)		#y of marker
	mT = CuArray{Float64,1}(undef, max_nmarker)		#T of marker

	#???
	staging = Array{Float64,1}(undef, max_npartcl)
	npartcl_d = CuArray{Int32,1}(undef, 1)
	npartcl_h = Array{Int32,1}(undef, 1)

	#small grid dimensions
	nxl = convert(Int64, nx / nl)
	nyl = convert(Int64, ny / nl)

	println("nx  - $nx")
	println("ny - $ny")
	println("nl - $nl")
	println(nxl)
	
	#small grid itself
	L = CuArray{Int32,1}(undef, nxl * nyl)

	#small grid on host
	L_host = Array{Int32,1}(undef, nxl * nyl)

	#???
	mfl = CuArray{Float64,1}(undef, nxl * nyl)


	#a and b of ellips for dikes
	dike_a = Array{Float64,1}(undef, ndikes_all)
	dike_b = Array{Float64,1}(undef, ndikes_all)

	#x and y coordinate of center
	dike_x = Array{Float64,1}(undef, ndikes_all)
	dike_y = Array{Float64,1}(undef, ndikes_all)

	#???
	dike_t = Array{Float64,1}(undef, ndikes_all)


	#NOTE:Dykes data upload takes time
	io = open("data/dikes.bin", "r");
	read!(io, dike_a)
	read!(io, dike_b)
	read!(io, dike_x)
	read!(io, dike_y)
	read!(io, dike_t)

	close(io)

	fid = h5open("data/particles.h5", "r")

	h_px = Array{Float64,1}(undef, max_npartcl)
	h_py = Array{Float64,1}(undef, max_npartcl)

	h_px = read(fid,"px")
	h_py = read(fid,"py")

	copyto!(px, h_px)
	copyto!(py, h_py)

	#???
	h_px_dikes = Array{Float64,1}(undef, np_dikes)
	h_py_dikes = Array{Float64,1}(undef, np_dikes)

	h_px_dikes = read(fid,"px_dikes")
	h_py_dikes = read(fid,"py_dikes")

	copyto!(px_dikes, h_px_dikes)
	copyto!(py_dikes, h_py_dikes)
	close(fid)

#=	
	#process markers
	fid = h5open("markers.h5", "r")

	obj = fid["0"]

	h_mx = Array{Float64,1}(undef, max_nmarker)
	h_my = Array{Float64,1}(undef, max_nmarker)
	h_mT = Array{Float64,1}(undef, max_nmarker)

	h_mx = read(obj, "mx")
	h_my = read(obj, "my")
	h_mT = read(obj, "mT")

	close(fid)
=#

	h_mx = Array{Float64,1}(undef, max_nmarker)
	h_my = Array{Float64,1}(undef, max_nmarker)
	h_mT = Array{Float64,1}(undef, max_nmarker)

	#copyto!(mx, h_mx)
	#copyto!(my, h_my)
	#copyto!(mT, h_mT)


	#reading initial grid

	NDIGITS = 5

	filename = "data/grid." * "0"^NDIGITS * "0" * ".h5"

	fid = h5open(filename, "r")
	T_h = read(fid, "T")
	copyto!(T, T_h)
	C_h = read(fid, "C")
	copyto!(C, C_h)
	close(fid)

	#auto tm_all = tic();

	global iSample = Int32(1)

	bar1 = "\n├──"
	bar2 = "\n\t ├──"
	#bar2 = "\xb3  \xc3\xc4\xc4";

	#@time begin
		#init
		#@time begin
			@printf("%s initialization			  ", bar1)
			pic_amount_tmp = pic_amount #???
			pic_amount = 1.0

			blockSize1D = 768
			gridSize1D = convert(Int64, floor((npartcl + blockSize1D - 1) / blockSize1D))

			#NOTE:
			#changing only pT
			#grid to particles interpolation
			#differene with cuda like 6.e-8 for some reason
			@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, px, py, pT, dx, dy, pic_amount, nx, ny, npartcl)

			gridSize1D = convert(
				Int64,
				floor((max_npartcl - npartcl + blockSize1D - 1) / blockSize1D),
			)

			pTs  = @view pT[npartcl+1:end];
			@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(pTs, T_magma, max_npartcl-npartcl);


			pPhs = @view pPh[npartcl+1:end];
			@cuda blocks = gridSize1D threads=blockSize1D init_particles_Ph(pPhs, 1, max_npartcl - npartcl);

			gridSize1D = Int64(floor((max_nmarker - nmarker + blockSize1D - 1) / blockSize1D))

			mTs = @view mT[nmarker+1:end];
			@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(mTs, T_magma, max_nmarker - nmarker);

			synchronize()

			pic_amount = pic_amount_tmp
		#end

		
		idike = 0
		global iSample = Int32(1)

		eruptionSteps = Vector{Int32}()

		#Main loop
		#for it ∈ 1:nt
		for it in 1:nt
			#action
			@printf("%s it = %d", bar1, it)
			is_eruption = false
			is_intrusion = (ndikes[it] > 0)
			#is_intrusion = false
			nerupt = 1;

			if (it % nerupt == 0)
				@time begin
					@printf("\n%s checking melt fraction   | ", bar2)

					blockSizel = (16, 32)
					gridSizel = (
						(nxl + blockSizel[1] - 1) ÷ blockSizel[1],
						(nyl + blockSizel[2] - 1) ÷ blockSizel[2],
					)

					#Усредняется по mf да mfl относительно содержания магмы и вмещающей породы
					#average<<<gridSizel, blockSizel>>>(mfl, T, C, nl, nx, ny);
					@cuda blocks = gridSizel threads=blockSizel average!(mfl, T, C, nl, nx, ny);

					synchronize()

					ccl(mfl, L, tsh, nxl, nyl)

					copyto!(L_host, L)

					volumes = Dict{Int32,Int32}(-1 => 0)

					#counting volumes
					for iy = 0:(nyl - 1)
						#taking into account only volumes higher then certain boundary
						if (iy * dy * nl < Ly_eruption)
							continue
						end
						for ix = 1:nxl
							if L_host[iy * nxl + ix] >= 0
								#WARN:for what?
								if haskey(volumes, L_host[iy * nxl + ix])
									volumes[L_host[iy * nxl + ix]] =
										volumes[L_host[iy * nxl + ix]] + 1
								else
									volumes[L_host[iy * nxl + ix]] = 0
									volumes[L_host[iy * nxl + ix]] =
										volumes[L_host[iy * nxl + ix]] + 1
								end
							end
						end
					end

					#maxVol - numbrer of cells
					maxVol = -1
					maxIdx = -1

					#searching for max vol
					for (idx, vol) in volumes
						if vol > maxVol
							maxVol = vol
							maxIdx = idx
						end
					end

				end

				dxl = dx * nl
				dyl = dy * nl

				#checking eruption criteria
				if (maxVol * dxl * dyl >= critVol[iSample])
					@printf("%s erupting %07d cells   | ", bar2, maxVol)
					@time begin

						cell_idx = CuArray{Int32,1}(undef, maxVol)
						cell_idx_host = Array{Int32,1}(undef, maxVol)


						next_idx = 0
						for idx = 0:(nxl * nyl)-1
							if L_host[idx+1] == maxIdx
								#if next_idx < maxVol
								next_idx = next_idx + 1
								cell_idx_host[next_idx] = idx
								#end
							end
						end

						copyto!(cell_idx, cell_idx_host)

						local blockSize1D = 512
						local gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D

						#advect particles
						@cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(px, py, cell_idx, gamma, dxl, dyl, npartcl, maxVol, nxl, nyl)
						synchronize()
			
						

						gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D
						#advect markers
						@cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(mx, my, cell_idx, gamma, dxl, dyl, nmarker, maxVol, nxl, nyl)
						synchronize()

						global iSample = iSample + 1

						is_eruption = true
						append!(eruptionSteps, it)
					end
				end
			end


			#processing intrusions of dike
			if (is_intrusion)
				@printf("%s inserting %02d dikes	   | ", bar2, ndikes[it])
				@time begin
					for i = 1:ndikes[it]
						idike = idike + 1

						blockSize1D = 512
						gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D

						#@printf("\nDebug\n")
						@cuda blocks = gridSize1D threads = blockSize1D advect_particles_intrusion(
							px,
							py,
							dike_a[idike],
							dike_b[idike],
							dike_x[idike],
							dike_y[idike],
							dike_t[idike],
							nu,
							G,
							ndikes[it],
							npartcl
						)

						dike_start = particle_edges[idike]
						dike_end = particle_edges[idike + 1]
						np_dike = dike_end - dike_start

						if (npartcl + np_dike > max_npartcl)
							@printf("ERROR: number of particles exceeds maximum value, increase capacity\n");
							return -1;
						end

						
					pxs = @view px[(npartcl+1):(npartcl+np_dike)];
					px_dikess = @view px_dikes[(dike_start+1):(dike_start+np_dike)];
					pys = @view py[(npartcl+1):(npartcl+np_dike)];
					py_dikess = @view py_dikes[(dike_start+1):(dike_start+np_dike)];
						

						copyto!(pxs, px_dikess)
						copyto!(pys, py_dikess)
					

						npartcl += np_dike

						gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D


						@cuda blocks = gridSize1D threads = blockSize1D advect_particles_intrusion(
							mx,
							my,
							dike_a[idike],
							dike_b[idike],
							dike_x[idike],
							dike_y[idike],
							dike_t[idike],
							nu,
							G,
							ndikes[it],
							nmarker,
						)

						synchronize()

						nmarker += marker_edges[idike + 1] - marker_edges[idike]

					end
				end
			end


			#if eruption or injection happend, taking into account their effcto on grid with p2g
			if (is_eruption || is_intrusion)
				@printf("%s p2g interpolation		| ", bar2)
				@time begin

					fill!(T, 0)
					fill!(C, 0)
					fill!(wts, 0)

					blockSize1D = 512
					gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D


					#p2g_project<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
					@cuda blocks = gridSize1D threads = blockSize1D p2g_project!(T, C, wts, px, py, pT, pPh, dx, dy, nx, ny, npartcl, npartcl0)
					synchronize()


					#p2g_weight<<<gridSize, blockSize>>>(ALL_ARGS);
					@cuda blocks = gridSize threads = blockSize p2g_weight!(T, C, wts, nx, ny)
					synchronize()


				end


				@printf("%s particle injection	   | ", bar2)

				@time begin


					blockSize1D = 512
					gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D

					#buf_arr_int = zeros(Int32, nx*ny)
					#copyto!(pcnt, buf_arr_int)
				
					#@printf("%s particle injection	   | ", bar2)
					#@device_code_warntype interactive=true @cuda blocks = gridSize threads=blockSize gpu_set_to_zero_2d!(pcnt, nx)

					#@printf("%s particle injection	   | ", bar2)
					fill!(pcnt, 0);

					#count particles
					@cuda blocks = gridSize1D threads=blockSize1D count_particles!(pcnt, px, py, dx, dy, nx, ny, npartcl);
					synchronize()

					#@printf("%s particle injection	   | ", bar2)
					#CUDA.allowscalar(true)
					
					npartcl_h[1] = npartcl
					copyto!(npartcl_d, npartcl_h);

					min_pcount = 2

					#inject particles where theit not enough
					@cuda blocks = gridSize threads=blockSize inject_particles(px, py, pT, pPh, npartcl_d, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl);
					synchronize()

					new_npartcl = npartcl
					#new_npartcl = npartcl_d[1]
					copyto!(npartcl_h, npartcl_d)

					new_npartcl = npartcl_h[1]

					#println(new_npartcl)

					#println(max_npartcl)

					#CUDA.allowscalar(false)

					if new_npartcl > max_npartcl
						fprintf(
							stderr,
							"ERROR: number of particles exceeds maximum value, increase capacity\n",
						)
						exit(EXIT_FAILURE)
					end

					if (new_npartcl > npartcl)
						@printf("(%03d) | ", new_npartcl - npartcl)
						npartcl = new_npartcl
					else
						@printf("(000) | ")
					end

				end
	
			end

			#solving heat equation
			#NOTE:difference like 2.e-1, mb make sense to fix it
			@time begin
				@printf("%s solving heat diffusion   | ", bar2)

				copyto!(T_old, T)
				for isub = 0:nsub-1
					@cuda blocks=gridSize[1],gridSize[2] threads=blockSize[1],blockSize[2] update_T!(T,  T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny);
					synchronize()
				end
			end


			#g2p interpolation
			@time begin
				@printf("%s g2p interpolation		| ", bar2)
				#particles g2p
				#println(npartcl)
				blockSize1D = 512
				gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
				@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, px, py, pT, dx, dy, pic_amount, nx, ny, npartcl)

				#markers g2p
				gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D
				pic_amount_tmp = pic_amount
				pic_amount = 1.0
				@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, mx, my, mT, dx, dy, pic_amount, nx, ny, nmarker);
				synchronize()
				pic_amount = pic_amount_tmp
			end

			
			if (it % nout == 0 || is_eruption)
				@time begin
					#@printf("\n%s writing debug results to disk  | ", bar2);
					#return 0;
					@printf("%s writing results to disk  | ", bar2)
					filename = "data/julia_grid." * string(it) * ".h5"

					small_mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl, max_nmarker, px, py, mx ,my, h_px_dikes,pcnt, mfl);
					#mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl, max_nmarker, px, py, mx ,my, h_px_dikes,pcnt, mfl);
					#=
					fid = h5open(filename, "w")

					write_h5(fid, "T", T, staging, nx * ny)
					write_h5(fid, "C", C, staging, nx * ny)

					if (is_eruption)
						write_h5(fid, "L", L, staging, nxl * nyl)
					end

					close(fid)
					=#
				end
			end
			#end

			#NOTE:writing markers to disk
			#not checked
			if(false)
				@time begin
				@printf("%s writing markers to disk  | ", bar2)
				filename = "markers.h5"
				fid = h5open(filename, "w")
				filename_gname = string(it) * "_gname"
				create_group(fid, filename_gname)
				g = fid[filename_gname]
				write_h5(g, "mx", mx, staging, nmarker)
				write_h5(g, "my", my, staging, nmarker)
				write_h5(g, "mT", mT, staging, nmarker)
				close(fid)
				end

			end

		end
		@printf("\nTotal time: ")

	fid = open("data/eruptions.bin", "w")
	write(fid, iSample)
	write(fid, eruptionSteps)
	close(fid)
	return 0
end
