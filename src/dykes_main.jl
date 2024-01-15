include("dykes_init.jl")
include("dykes_funcs.jl")

import Random

function main()

	#Initialization of inner random
	Random.seed!(1234)

	#TODO:Настроить фильтр

	#TODO:Настроить девайс если не выбран
	#
	#Print properties
	print_gpu_properties()

	dpa = Array{Float64,1}(undef, 18)	#array of double values from matlab script
	ipa = Array{Int32,1}(undef, 12)		#array of int values from matlab script

	#Initialization of main variables
	Lx = 0.0				#X size of researched area (m)
	Ly = 0.0				#Y size of researched area (m)
	lam_r_rhoCp = 0.0		#Thermal conductivity of rock/(density*specific heat capacity)
	lam_m_rhoCp = 0.0		#Thermal conductivity of magma/(density*specific heat capacity)
	L_Cp = 0.0				#dT/Ste, Ste = dT/(Lheat/Cp); L_heat/Cp
	T_top = 0.0				#Temperature on the top of area (C)
	T_bot = 0.0				#Temperature at the bottom of the area (C)
	T_magma = 0.0			#Magma instrusion temperature(C)
	tsh = 0.0				#nondimensonal 0.75
	gamma = 0.0				#nondimensional 0.1
	Ly_eruption = 0.0		#? 2000, m
	nu = 0.0				#Poisson ratio of rock
	G = 0.0					#E/(2*(1+nu));
	dt = 0.0				#time step
	dx = 0.0				#X dimension step
	dy = 0.0				#Y dimension step
	eiter = 0.0				#epsilon?
	pic_amount = 0.0		#?, 0.05

	pmlt = 0				#unused
	nx = 0					#Resolution for X dimension
	ny = 0					#Resolution for Y dimension
	nl = 0					#?
	nt = 0					#?
	niter = 0				#?
	nout = 0				#?
	nsub = 0				#?
	nerupt = 0				#number of eruptions
	npartcl = 0				#number of particles
	nmarker = 0				#number of markers
	nSample = 0				#size of a Sample, 1000

	#char filename[1024];
	filename = Array{Char,1}(undef, 1024)


	io = open("pa.bin", "r")
	read!(io, dpa)
	read!(io, ipa)

	ipar = 1								#index to read parameters
	Lx, ipar = read_par(dpa, ipar)			#x length of area
	Ly, ipar = read_par(dpa, ipar)			#y length of area
	lam_r_rhoCp, ipar = read_par(dpa, ipar)	#a few coefficients multiplied
	lam_m_rhoCp, ipar = read_par(dpa, ipar)	# a few coefficients multipled
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

	critVol = Array{Float64,1}(undef, nSample)
	read!(io, critVol)

	ndikes = Array{Int32,1}(undef, nt)
	read!(io, ndikes)

	ndikes_all = 0

	for istep in 1:nt
		ndikes_all = ndikes_all + ndikes[istep]
	end

	println("ndikes_all")
	println(ndikes_all)

	particle_edges = Array{Int32,1}(undef, ndikes_all + 1)
	read!(io, particle_edges)

	marker_edges = Array{Int32,1}(undef, ndikes_all + 1)
	read!(io, marker_edges)

	close(io)

	cap_frac = 1.5
	npartcl0 = npartcl
	max_npartcl = convert(Int64, npartcl * cap_frac) + particle_edges[ndikes_all+1]

	println(max_npartcl)

	nmarker0 = nmarker

	max_nmarker = nmarker + marker_edges[ndikes_all+1]


	#dim3 blockSize(16, 32);
	# dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

	T = CuArray{Float64,2}(undef, nx, ny)
	T_old = CuArray{Float64,2}(undef, nx, ny)
	C = CuArray{Float64,2}(undef, nx, ny)
	wts = CuArray{Float64,2}(undef, nx, ny)
	pcnt = CuArray{Int32,2}(undef, nx, ny)

	a = CuArray{Float64}(undef, (1, 2))

	px = CuArray{Float64}(undef, max_npartcl)			#x coordinate of particle
	py = CuArray{Float64}(undef, max_npartcl)
	pT = CuArray{Float64}(undef, max_npartcl)
	pPh = CuArray{Int8}(undef, max_npartcl)

	np_dikes = particle_edges[ndikes_all+1]

	px_dikes = CuArray{Float64,1}(undef, np_dikes)
	py_dikes = CuArray{Float64,1}(undef, np_dikes)

	mx = CuArray{Float64,1}(undef, max_nmarker)
	my = CuArray{Float64,1}(undef, max_nmarker)
	mT = CuArray{Float64,1}(undef, max_nmarker)

	staging = Array{Float64,1}(undef, max_npartcl)
	npartcl_d = Array{Int32,1}(undef, 1)


	nxl = convert(Int64, nx / nl)
	nyl = convert(Int64, ny / nl)

	println("nx  - $nx")
	println("ny - $ny")
	println("nl - $nl")
	println(nxl)
	L = CuArray{Int32,1}(undef, nxl * nyl)

	L_host = Array{Int32,1}(undef, nxl * nyl)

	mfl = CuArray{Float64,2}(undef, nxl, nyl)

	dike_a = Array{Float64,1}(undef, ndikes_all)
	dike_b = Array{Float64,1}(undef, ndikes_all)
	dike_x = Array{Float64,1}(undef, ndikes_all)
	dike_y = Array{Float64,1}(undef, ndikes_all)
	dike_t = Array{Float64,1}(undef, ndikes_all)
	#NOTE:Dykes data upload takes time

	io = open("dikes.bin", "r");
	read!(io, dike_a)
	read!(io, dike_b)
	read!(io, dike_x)
	read!(io, dike_y)
	read!(io, dike_t)

	close(io)

	fid = h5open("particles.h5", "r")

	h_px = Array{Float64,1}(undef, max_npartcl)
	h_py = Array{Float64,1}(undef, max_npartcl)

	h_px = read(fid,"px")
	h_py = read(fid,"py")

	copyto!(px, h_px)
	copyto!(py, h_py)

	h_px_dikes = Array{Float64,1}(undef, np_dikes)
	h_py_dikes = Array{Float64,1}(undef, np_dikes)

	h_px_dikes = read(fid,"px_dikes")
	h_py_dikes = read(fid,"py_dikes")

	copyto!(px_dikes, h_px_dikes)
	copyto!(py_dikes, h_py_dikes)
	close(fid)

	fid = h5open("markers.h5", "r")

	obj = fid["0"]
	read(obj, "mx")
	read(obj, "my")
	read(obj, "mT")

	close(fid)

	NDIGITS = 4

	filename = "grid." * "0"^NDIGITS * "0" * ".h5"

	fid = h5open(filename, "r")
	T_h = read(fid, "T")
	copyto!(T_h, T)
	C_h = read(fid, "C")
	copyto!(C_h, C)
	close(fid)

	#auto tm_all = tic();

	bar1 = "├──"
	bar2 = "\t ├──"
	#bar2 = "\xb3  \xc3\xc4\xc4";

	@time begin
		#init
		@time begin
			@printf("%s initialization			  ", bar1)
			pic_amount_tmp = pic_amount
			pic_amount = 1.0

			blockSize1D = 768
			gridSize1D = convert(Int64, floor((npartcl + blockSize1D - 1) / blockSize1D))

		#__________________________________________________________
			#@cuda blocks = gridSize1D threads=blockSize1D g2p(@ALL_ARGS())

			#kekw = idc(2, 4, nx)
			#println("$kekw");

			#changing only pT
			@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)

			@printf("%s writing debug results to disk  | ", bar2)
			mailbox_out("julia_out.h5",T,pT, C,staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl);
			return 0;
		#__________________________________________________________
			gridSize1D = convert(
				Int64,
				floor((max_npartcl - npartcl + blockSize1D - 1) / blockSize1D),
			)

			#@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(pT, T_magma, max_npartcl, npartcl);
			#@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(pPh, 1, max_npartcl, npartcl);
			gridSize1D = (max_nmarker - nmarker + blockSize1D - 1) / blockSize1D
			#@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(mT, T_magma, max_nmarker, nmarker);
			synchronize()

			pic_amount = pic_amount_tmp
		end

		
		idike = 1
		iSample = Int32(1)

		eruptionSteps = Vector{Int32}()

		#Main loop
		for it ∈ 1:nt
			#action
			@printf("%s it = %d", bar1, it)
			is_eruption = false
			is_intrusion = (ndikes[it] > 0)

			if (it % nerupt == 0)
				@time begin
					@printf("\n%s checking melt fraction   | ", bar2)

					blockSizel = (16, 32)
					gridSizel = (
						(nxl + blockSizel[1] - 1) ÷ blockSizel[1],
						(nyl + blockSizel[2] - 1) ÷ blockSizel[2],
					)

					#TODO:Wrong functionns with reologyy inside
					@cuda blocks = gridSizel threads=blockSizel average(mfl, T, C, nl, nx, ny);

					#Усредняется по температуре относительно содержания магмы и вмещающей породы
					#для уменьшенной сетки
					#average<<<gridSizel, blockSizel>>>(mfl, T, C, nl, nx, ny);
					synchronize()

					#checked
					ccl(mfl, L, tsh, nxl, nyl)

					copyto!(L, L_host)
					#cudaMemcpy(L_host, L, SIZE_2D(nxl, nyl, int), cudaMemcpyDeviceToHost)

					volumes = Dict{Int32,Int32}(-1 => 0)

					#println("\n");
					#println(volumes);

					for iy = 0:(nyl - 1)
						if (iy * dy * nl < Ly_eruption)
							continue
						end
						for ix = 1:nxl
							if L_host[iy * nxl + ix] >= 0
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

					maxVol = -1
					maxIdx = -1

					for (idx, vol) in volumes
						if vol > maxVol
							maxVol = vol
							maxIdx = idx
						end
					end

				end

				dxl = dx * nl
				dyl = dy * nl

				if (maxVol * dxl * dyl >= critVol[iSample])
				#if (true)
					@printf("%s erupting %07d cells   | ", bar2, maxVol)
					@time begin

						cell_idx = CuArray{Int32,1}(undef, maxVol)
						cell_idx_host = Array{Int32,1}(undef, maxVol)
						next_idx = 0



						for idx = 1:(nxl * nyl)
							if L_host[idx] == maxIdx
								if next_idx < maxVol
									next_idx = next_idx + 1
									cell_idx_host[next_idx] = idx
								end
							end
						end

						copyto!(cell_idx, cell_idx_host)

						blockSize1D = 512
						gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
						#advect_particles_eruption<<<gridSize1D, blockSize1D>>>(px, py, cell_idx, gamma, dxl, dyl, npartcl, maxVol, nxl, nyl);
						#=
						@cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(
							px,
							py,
							cell_idx,
							gamma,
							dxl,
							dyl,
							npartcl,
							5,
							nxl,
							nyl,
						)
						synchronize()
						=#
						gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D

						#advect_particles_eruption<<<gridSize1D, blockSize1D>>>(mx, my, cell_idx, gamma, dxl, dyl, nmarker, maxVol, nxl, nyl);
						#=
						@cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(
							mx,
							my,
							cell_idx,
							gamma,
							dxl,
							dyl,
							nmarker,
							5,
							nxl,
							nyl,
						)
						synchronize()
							=#
						iSample = iSample + 1

						is_eruption = true
						append!(eruptionSteps, it)

					end
				end
			end

			if (is_intrusion)
				@printf("%s inserting %02d dikes	   | ", bar2, ndikes[it])
				@time begin
					for i = 1:ndikes[it]
						idike = idike + 1
						blockSize1D = 512
						gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
						#advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(px, py, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
								 #									ndikes[it - 1], npartcl);
						#
						#=
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
							npartcl,
						)
						=#
						dike_start = particle_edges[idike]
						dike_end = particle_edges[idike + 1]
						np_dike = dike_end - dike_start

						if (npartcl + np_dike > max_npartcl)
							fprintf(
								stderr,
								"ERROR: number of particles exceeds maximum value, increase capacity\n",
							)
							exit(EXIT_FAILURE)
						end


						#copyto!(px[npartcl], px_dikes[dike_start])
						#px_dikes[dike_start] = px[npartcl]
						#py_dikes[dike_start] = py[npartcl]
						#copyto!(py[npartcl], py_dikes[dike_start])
						npartcl += np_dike

						gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D


						#advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(mx, my, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
						#ndikes[it - 1], nmarker);
						#=
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
						=#
						nmarker += marker_edges[idike + 1] - marker_edges[idike]
					end
				end
			end



			if (is_eruption || is_intrusion)
				@printf("%s p2g interpolation		| ", bar2)
				@time begin

					#fill!(T, nx*ny)
					#T = fill(Float64, nx*ny)
					T = CUDA.zeros(Float64, nx * ny)
					C = CUDA.zeros(Float64, nx * ny)
					wts = CUDA.zeros(Float64, nx * ny)

					blockSize1D = 768
					gridSize1D = (npartcl + blockSize1D - 1) / blockSize1D

					#p2g_project<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
					synchronize()
					#p2g_weight<<<gridSize, blockSize>>>(ALL_ARGS);
					synchronize()

				end

				@printf("%s particle injection	   | ", bar2)
				@time begin


					blockSize1D = 512
					gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
					pcnt = CUDA.zeros(Float64, nx * ny)
					#count_particles<<<gridSize1D, blockSize1D>>>(pcnt, px, py, dx, dy, nx, ny, npartcl);
					#@cuda blocks = gridSize1D threads=blockSize1D count_particles(pcnt, px, py, dx, dy, nx, ny, npartcl);
					synchronize()

					npartcl_d[1] = npartcl
					min_pcount = 2
					#inject_particles<<<gridSize, blockSize>>>(px, py, pT, pPh, npartcl_d, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl);
					synchronize()
					new_npartcl = npartcl

					npartcl = npartcl_d[1]

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




			@time begin
				@printf("%s solving heat diffusion   | ", bar2)

				copyto!(T, T_old)
				for isub = 0:nsub
					#update_T<<<gridSize, blockSize>>>(ALL_ARGS);
					synchronize()
				end
			end

			@time begin
				@printf("%s g2p interpolation		| ", bar2)
				#particles g2p
				blockSize1D = 1024
				gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
				#g2p<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
				#@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)

				#markers g2p
				gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D
				pic_amount_tmp = pic_amount
				pic_amount = 1.0
				#g2p<<<gridSize1D, blockSize1D>>>(T, T_old, C, wts, mx, my, mT, NULL, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, nmarker,
				#			 nmarker0);
				#@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)
				synchronize()
				pic_amount = pic_amount_tmp

			end


			@printf("%s writing debug results to disk  | ", bar2)
			mailbox_out("julia_out.h5",T,pT, C,staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl);
			return 0;

			if (it % nout == 0 || is_eruption)
				#auto tm = tic();
				@time begin
					@printf("%s writing results to disk  | ", bar2)
					filename = "grid." * string(it) * ".h5"
					fid = h5open(filename, "w")

					write_h5(fid, "T", T, staging, nx * ny)
					write_h5(fid, "C", C, staging, nx * ny)

					if (is_eruption)
						write_h5(fid, "L", L, staging, nxl * nyl)
					end

					close(fid)
				end
			end


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

			break
		end
		@printf("\nTotal time: ")
	end #for_time

	fid = open("eruptions.bin", "w")
	write(fid, iSample)
	write(fid, eruptionSteps)
	close(fid)

end
