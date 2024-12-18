using Dates
include("dykes_init.jl")

using PlotlyJS, CSV, DataFrames
using Plots

function main()
	# global_time = @elapsed begin
	# loop_max = 100000
	# 	for i in 1:2
	# 		time_of_loop = @elapsed begin
	# 			println("capre diem")
	# 		end
	#
	# 		println("time_of_loop")
	# 		println(time_of_loop)
	# 		println(i)
	# 		println(Second(Int64(floor(time_of_loop/(i/Float64(loop_max))))))
	# 		println(Time(0)+Second(Int64(10000)))
	# 		str_time_left = floor(time_of_loop/(i/Float64(loop_max)))
	# 		println("str_time_left")
	# 		println(str_time_left)
	# 	end
	# end
	
	
	dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
	ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script

	io = open(data_folder * "pa.bin", "r")
	read!(io, dpa)
	read!(io, ipa)

	ipar = 1
	Lx, ipar = read_par(dpa, ipar)
	Ly, ipar = read_par(dpa, ipar)
	lam_r_rhoCp, ipar = read_par(dpa, ipar)
	lam_m_rhoCp, ipar = read_par(dpa, ipar)
	L_Cp, ipar = read_par(dpa, ipar)
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
	tfin, ipar = read_par(dpa, ipar)

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
	ndikes = Array{Int32,1}(undef, nt)#number of dykes intruded on n-th time step
	read!(io, ndikes)

	ndikes_all = 0

	#count all dykes
	for istep in 1:nt
		ndikes_all = ndikes_all + ndikes[istep]
	end

	#array which describes amount of particles in new dyke
	particle_edges = Array{Int32,1}(undef, ndikes_all + 1)
	read!(io, particle_edges)

	marker_edges = Array{Int32,1}(undef, ndikes_all + 1)
	read!(io, marker_edges)

	close(io)

	cap_frac = 3  #value to spcify how much particles we allow to inject in runtime
	npartcl0 = npartcl #initial amount of particles
	max_npartcl = convert(Int64, npartcl * cap_frac) + particle_edges[ndikes_all+1] #???#count max particles
	println(npartcl)
	println(particle_edges[ndikes_all+1])
	println("max_npartcl")
	println(max_npartcl)
	nmarker0 = nmarker


	#	max_nmarker = nmarker + marker_edges[ndikes_all+1]



	np_dikes = particle_edges[ndikes_all+1]#number of particles in each dike during intrusion

	fid = h5open(data_folder * "particles.h5", "r")

	h_px = Array{Float64,1}(undef, max_npartcl)
	h_py = Array{Float64,1}(undef, max_npartcl)

	h_px = read(fid, "px")
	h_py = read(fid, "py")


	h_px_dikes = Array{Float64,1}(undef, np_dikes)
	h_py_dikes = Array{Float64,1}(undef, np_dikes)

	h_px_dikes = read(fid, "px_dikes")
	h_py_dikes = read(fid, "py_dikes")

	#PlotlyJS.scatter([1,2,3],[4,5,6])

	close(fid)
	#Plots.covellipse!([0,2], [2 1; 1 4], n_std=2, aspect_ratio=1, label="cov1")
	d2d_limit =np_dikes 
	d2d_limit_gap = 100
	Plots.scatter(markersize = 0.1, h_px_dikes[1:d2d_limit_gap:d2d_limit],h_py_dikes[1:d2d_limit_gap:d2d_limit], xlimit = [1, 20000], ylimit = [1, 20000])





	# pts = Plots.partialcircle(0, 2π, 100, 0.1)
	# x, y = Plots.unzip(pts)
	# x = 1.5x .+ 0.7
	# y .+= 1.3
	# pts = collect(zip(x, y))
	#
	# plot!(Plots.Shape(x, y), c = :yellow)
		#Plots.covellipse([0,2], [2 1; 1 4], n_std=2, aspect_ratio=1, label="cov1")

		#	PlotlyJS.plot([
		# test_fig = PlotlyJS.scatter(x=h_px_dikes, y=h_py_dikes, mode="markers", name="markers")
		#    PlotlyJS.plot([test_fig])
		#	])
end


function d2d_test_write()

	vp = VarParams()			#scalar params
	gp = GridParams()			#array params

	vp.Lx = 8484
	gp.critVol = [1.0,2.0];
	#wts = CuArray{Float64,1}()
	gp.wts = CuArray{Float64,1}(undef, 2)
	copyto!(gp.wts,gp.critVol)
	filename_donwload = "tesst_6.h5"

	#filename_donwload = @sprintf("d2d_snapshot_%d_%s.hdf5",vp.it, Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))
	#filename_donwload = @sprintf("d2d_config_%s.hdf5",dates.format(now(), "yyyy_mm_dd_hh_mm_ss"))

	if isfile(filename_donwload)
		rm(filename_donwload)
	end

	fid = h5open(filename_donwload, "w")

	for n in fieldnames(typeof(vp))
		println(getfield(vp,n))
		write(fid, string(n), getfield(vp,n))
	end

	for n in fieldnames(typeof(gp))
		if(getfield(gp,n) isa CuArray)
			d2d_cu_type = eltype(getfield(gp,n))

			nn::Array{d2d_cu_type,1} = Array{d2d_cu_type,1}(undef, size(getfield(gp,n))[1]);
			copyto!(nn, getfield(gp,n))

			println(getfield(gp,n))
			write(fid, string(n), nn)
			println("sucess!!")
		else
			println(getfield(gp,n))
			write(fid, string(n), getfield(gp,n))
		end
	end

	#return dict("content" => vector{uint8}(fid), "filename" => filename_donwload) # get a byte vector to send, e.g., using http, mqtt or similar.
	println("snapshot saved to " * filename_donwload)
	#log_to_buffer("snapshot saved to " *  filename_donwload)
	close(fid)

end

function d2d_test_read()

#	filename = @sprintf("test_%s.hdf5",Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))
#	filename = "test_3.hdf5"

	filename_donwload = "tesst_6.h5"

	vp = VarParams()			#scalar params
	gp = GridParams()			#array params

	fid = h5open(filename_donwload, "r")
	#init_vp = InitVarParams()	#params for generate random

	#init_vp.critVol = global_EruptionVolumesVec
	#init_vp.critVolTime = global_EruptionTimesVec

	for n in fieldnames(typeof(vp))
		setfield!(vp, n, read(fid, string(n)))
		println(getfield(vp,n))
	end

	for n in fieldnames(typeof(gp))
		if(getfield(gp,n) isa CuArray)
			d2d_cu_type = eltype(getfield(gp,n))
			nn::CuArray{d2d_cu_type,1} = CuArray{d2d_cu_type,1}(undef, size(getfield(gp,n))[1]);
			#copyto!(nn, read(fid, string(n)))
			nn = read(fid, string(n))
			setfield!(gp, n, nn)
			#copyto!(getfield(gp,n), read(fid, string(n)))
			#copyto!(getfield(gp,n), read(fid, string(n)))
			#write(fid, string(n), nn)
			
			println("GPU")
			println(getfield(gp,n))
		else
			#read(fid, string(n), getfield(gp,n))
			

			#d2d_cu_type = eltype(getfield(gp,n))
			#nnn::Array{d2d_cu_type,1} = Array{d2d_cu_type,1}(undef, size(getfield(gp,n))[1]);

			#nnn = read(fid, string(n))
			setfield!(gp, n,  read(fid, string(n)))
			println("CPU")
			println(getfield(gp,n))
		end
	end
	
	println("success?")


	close(fid)
end

function d2d_test()

	# wts::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);
	# if(wts isa CuArray)
	# 	println("true")
	# else
	# 	println("false")
	# end
end



	# people = [Dict("name"=>"CoolGuy", "company"=>"tech") for i=1:1000]
	# companies = [Dict("name"=>"CoolTech", "address"=>"Bay Area") for i=1:100]
	#
	# data = Dict("people"=>people, "companies"=>companies)
	#
	# open("foo.json","w") do f
	# 	JSON.print(f, data)
	# end


# function area1()
# 	trace1 = scatter(;x=1:4, y=[0, 2, 3, 5], fill="tozeroy")
# 	trace2 = scatter(;x=1:4, y=[3, 5, 1, 7], fill="tonexty")
# 	plot([trace1, trace2])
# end
# area1()
