include("dykes_init.jl")
include("dykes_funcs.jl")


function dykes_graph()

    #fid = h5open(data_folder * "julia_grid.120000.h5", "r")
    #Lx = read(fid, "Lx")
    #Ly = read(fid, "Ly")
    #close(fid)

    #fid = h5open(data_folder * "julia_grid.120000.h5", "r")

    dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
    ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script

    io = open(data_folder*"pa.bin", "r")
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

    close(io)

	#Lx = 4000
	#Lx = 5000
    xs = 0:dx:Lx
    ys = 0:dy:Ly

#    fid = h5open(data_folder * "julia_grid.12001.h5", "r")
    fid = h5open(data_folder * "julia_grid.3333.h5", "r")
    T = read(fid, "T")
    C = read(fid, "C")
    close(fid)

	l = @layout [ [grid(1,2)] 
					a]

	#campri_calc = Array{Int32,1}(undef, 4)#array of int values from matlab script
	campri_calc = Int32[10000, 20000, 25000, 26000]
    #write(data_folder*"eruptions.bin", campri_calc)

	println(typeof(campri_calc))
	file = open(data_folder*"eruptions.bin", "w")
	write(file, campri_calc)
	close(file)

	fz = filesize(data_folder*"eruptions.bin")
	fz_int = Int32(floor(fz/sizeof(Int32)))

	println(fz/sizeof(Int32))
	if(fz_int <= 1)
		print("No eruptions!!!")
		return
	end
	campri_calc = Array{Int32,1}(undef, fz_int)#array of int values from matlab script
	read!(data_folder*"eruptions.bin", campri_calc)

	tyear = 365 * 24 * 3600		#seconds in year
	tfin = (tfin/tyear)/1.e3
	campri_calc = -(1 .- campri_calc./nt) .* (tfin)
	println(campri_calc)
	display(campri_calc)
	campri_real = -vcat(39.8, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 11.5, 11, 10.6, 9.6, 9.3, 5.1, 4.9, 4.5, 4.3, 4.2, 4.2, 4.2, 4.1, 3.9, 0.5);

	p = scatter(campri_real, zeros(length(campri_real)), markersize=7, 
        markershape=:circle, color = :red, legend=true, 
			 framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="real campri", xlimits=(min(minimum(campri_real),minimum(campri_calc))-10, 0))

	p = scatter!(campri_calc, zeros(length(campri_calc)), markersize=4, 
        markershape=:circle, color = :blue, legend=true, 
        framestyle=:origin, yaxis=false, grid=false, aspect_ratio=1.0, label="calc campri", markeralpha = 0.5)

	T = reshape(T,(length(xs), length(ys)))
	C = reshape(C,(length(xs), length(ys)))
	p1 = plot(heatmap(ys, xs, transpose(T)), heatmap(ys, xs, transpose(C)), p, layout = l)

	display(p1)
end
