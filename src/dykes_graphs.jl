

function dykes_graph()
    #fid = h5open(data_folder * "julia_grid.120000.h5", "r")
    #Lx = read(fid, "Lx")
    #Ly = read(fid, "Ly")
    #close(fid)

    #fid = h5open(data_folder * "julia_grid.120000.h5", "r")

    dpa = Array{Float64,1}(undef, 18)#array of double values from matlab script
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
    fid = h5open(data_folder * "grid.00000.h5", "r")
    T = read(fid, "T")
	
    close(fid)

    #x, y = meshgrid(xs, ys)
	T = reshape(T,(length(xs), length(ys)))
	@infiltrate
	heatmap(ys, xs, T)
end
