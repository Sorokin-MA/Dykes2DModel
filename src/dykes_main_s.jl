#main jl file made with structs
include("dykes_init.jl")
include("dykes_structs.jl")
include("dykes_funcs.jl")



function main_test()
    #Initialization of inner random
    Random.seed!(1234)
    checker = Array{Float64}(undef, 1)
    #TODO:Настроить фильтр
    #TODO:Настроить девайс если не выбран

    #print_gpu_properties()

    gp = GridParams()
    vp = VarParams()

    #reading params from hdf5 files
    @printf("%s reading params			  ", bar1)
    read_params(gp, vp)

    #initialisation of T and Ph variables
    @printf("%s initialization			  ", bar1)
    init(gp, vp)

    filename = Array{Char,1}(undef, 1024)
    eruptionSteps = Vector{Int32}()
    eruption_counter::Int64 = 1
    eruption_vol = Vector{Float64}()
    
    #main loop
    for it in 1:vp.nt
        @printf("%s it = %d", bar1, it)
		vp.is_eruption = false
        eruption_counter = eruption_counter - 1
        is_intrusion = (gp.ndikes[it] > 0)
        nerupt = 1

        #checking eruption criteria and advect particles if eruption
        if (it % nerupt == 0)
            #calculating maxVol
            maxVol, maxIdx = check_melt_fracton(gp, vp)

            if maxVol == -1
                return 0
            end

            dxl = vp.dx * vp.nl
            dyl = vp.dy * vp.nl
            real_vol = (maxVol * (dxl * dyl) / 1.e9) * 1.e4 * vp.gamma
            @printf("%s accomulated %06f km^3| ", bar2, (maxVol * (dxl * dyl) / 1.e9) * 1.e4 * vp.gamma)

            if (maxVol * dxl * dyl >= gp.critVol[vp.iSample] &&  eruption_counter <=0)
                @printf("%s erupting %07d cells   | ", bar2, maxVol)

                @printf("%s writing results to disk  | ", bar2)
                filename = data_folder * "julia_grid." * string(it) * "before_eruption" * ".h5"

                small_mailbox_out(filename, gp.T, gp.pT, gp.C, gp.mT, gp.staging, gp.L, vp.nx, vp.ny, vp.nxl, vp.nyl, vp.max_npartcl, vp.max_nmarker, gp.px, gp.py, gp.mx, gp.my, gp.h_px_dikes, gp.pcnt, gp.mfl)
                #mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl, max_nmarker, px, py, mx ,my, h_px_dikes,pcnt, mfl);

                append!(eruption_vol, real_vol)

                eruption_advection(gp, vp, maxVol, maxIdx, it)
                println("vp.is_eruption")
                println(vp.is_eruption)
                eruption_counter = 10
            end
        end


        #processing intrusions of dikes
        if (is_intrusion)
            @printf("%s inserting %02d dikes	   | ", bar2, gp.ndikes[it])
            inserting_dykes(gp, vp, it)
        end


        #if eruption or injection happend, taking into account their effcto on grid with p2g
        if (vp.is_eruption || is_intrusion)
            @printf("%s p2g interpolation		| ", bar2)
            p2g_interpolation(gp, vp)


            @printf("%s particle injection	   | ", bar2)
            particles_injection(gp, vp)
        end


        #solving heat equation
        #NOTE:difference like 2.e-1, mb make sense to fix it
        @time begin
            @printf("%s solving heat diffusion   | ", bar2)

            blockSize = (16, 32)
            gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

            copyto!(gp.T_old, gp.T)
            for isub = 0:vp.nsub-1
                @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] update_T!(gp.T, gp.T_old, vp.T_top, vp.T_bot, gp.C, vp.lam_r_rhoCp, vp.lam_m_rhoCp, vp.L_Cp, vp.dx, vp.dy, vp.dt, vp.nx, vp.ny)
                synchronize()
            end

            #check for calculation explosion
            T_for_chek = @view gp.T[1]
            copyto!(checker, T_for_chek)
            if isnan(checker[1])
                println("EXPLOSION!!!")
                return -1
            end
        end


        #g2p interpolation
        @time begin
            @printf("%s g2p interpolation		| ", bar2)
            blockSize1D = 512
            gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D
            @cuda blocks = gridSize1D threads = blockSize1D g2p!(gp.T, gp.T_old, gp.px, gp.py, gp.pT, vp.dx, vp.dy, vp.pic_amount, vp.nx, vp.ny, vp.npartcl)

            gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D
            pic_amount_tmp = vp.pic_amount
            pic_amount = 1.0
            @cuda blocks = gridSize1D threads = blockSize1D g2p!(gp.T, gp.T_old, gp.mx, gp.my, gp.mT, vp.dx, vp.dy, vp.pic_amount, vp.nx, vp.ny, vp.nmarker)
            synchronize()
            vp.pic_amount = pic_amount_tmp
        end


        #mailbox output
        if (it % vp.nout == 0 || vp.is_eruption)
            @time begin
                @printf("%s writing results to disk  | ", bar2)
                filename = data_folder * "julia_grid." * string(it) * ".h5"

                small_mailbox_out(filename, gp.T, gp.pT, gp.C, gp.mT, gp.staging, gp.L, vp.nx, vp.ny, vp.nxl, vp.nyl, vp.max_npartcl, vp.max_nmarker, gp.px, gp.py, gp.mx, gp.my, gp.h_px_dikes, gp.pcnt, gp.mfl)
                #mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl, max_nmarker, px, py, mx ,my, h_px_dikes,pcnt, mfl);
            end
        end
    end

    @printf("%s writing results to disk  | ", bar2)
    filename = data_folder * "julia_grid." * string(vp.nt + 1) * ".h5"
    small_mailbox_out(filename, gp.T, gp.pT, gp.C, gp.mT, gp.staging, gp.L, vp.nx, vp.ny, vp.nxl, vp.nyl, vp.max_npartcl, vp.max_nmarker, gp.px, gp.py, gp.mx, gp.my, gp.h_px_dikes, gp.pcnt, gp.mfl)

    @printf("\nTotal time: ")

    fid = open(data_folder * "eruptions.bin", "w")
    write(fid, vp.iSample)
    write(fid, gp.eruptionSteps)
    close(fid)
    
    fid = open(data_folder * "eruptions_vol.bin", "w")
    write(fid, vp.iSample)
    write(fid, eruption_vol)
    close(fid)

    return 0
end
