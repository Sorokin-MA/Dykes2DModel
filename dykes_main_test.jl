#main
    
    include("dykes_funcs.jl")
    #Checking if chosen filter for HDF5 lib is available 
    if !HDF5.Filters.isavailable(@H5Z_FILTER_BLOSC)
        fprintf(stderr, "Error: blosc filter is not available\n")
        exit(1)
    end

    printDeviceProperties(@GPU_ID)
    device!(@GPU_ID)
    #TODO:figure out what is this optimisation
    #cudaDeviceReset()
    #cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)

    dpa = Array{Float64,1}(undef, 18)
    ipa = Array{Int32,1}(undef, 12)

    Lx, Ly, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, T_magma, tsh, gamma, Ly_eruption, nu, G, dt, dx, dy, eiter, pic_amount = [0.0 for _ = 1:18]
    pmlt, nx, ny, nl, nt, niter, nout, nsub, nerupt, npartcl, nmarker, nSample =  [0 for _ = 1:12]

    filename = fill(' ', 1024)

    h = open("pa.bin", "r")
    read!(h, dpa)
    read!(h, ipa)

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

    critVol = Array{Float64,1}(undef, nSample)
    read!(h, critVol)

    ndikes = Array{Int32,1}(undef, nt)
    read!(h, ndikes)

    ndikes_all = 0


    for istep in 1:nt
        global ndikes_all += ndikes[istep]
    end


    particle_edges = Array{Int32,1}(undef, ndikes_all + 1)
    read!(h, particle_edges)

    marker_edges = Array{Int32,1}(undef, ndikes_all + 1)
    read!(h, marker_edges)

    close(h)

    cap_frac = 1.5

    npartcl0 = npartcl
    max_npartcl = Int(npartcl * cap_frac) + particle_edges[ndikes_all]

    nmarker0 = nmarker
    max_nmarker = nmarker + marker_edges[ndikes_all]

    blockSize = (16, 32)
    gridSize = ((nx + blockSize[1] - 1) ÷ blockSize[1], (ny + blockSize[2] - 1) ÷ blockSize[2])

    T = CUDA.CuArray{Float64}(undef, nx, ny)
    T_old = CUDA.CuArray{Float64}(undef, nx, ny)
    C = CUDA.CuArray{Float64}(undef, nx, ny)
    wts = CUDA.CuArray{Float64}(undef, nx, ny)
    pcnt = CUDA.CuArray{Int}(undef, nx, ny)

    px = CUDA.CuArray{Float64}(undef, max_npartcl)
    py = CUDA.CuArray{Float64}(undef, max_npartcl)
    pT = CUDA.CuArray{Float64}(undef, max_npartcl)
    pPh = CUDA.CuArray{Int8}(undef, max_npartcl)

    np_dikes = particle_edges[ndikes_all]
    px_dikes = CUDA.CuArray{Float64}(undef, np_dikes)
    py_dikes = CUDA.CuArray{Float64}(undef, np_dikes)

    mx = CUDA.CuArray{Float64}(undef, max_nmarker)
    my = CUDA.CuArray{Float64}(undef, max_nmarker)
    mT = CUDA.CuArray{Float64}(undef, max_nmarker)

    staging = CUDA.malloc(SIZE_1D(max_npartcl, Float64))
    npartcl_d = CUDA.malloc(sizeof(Int))

    nxl = nx ÷ nl
    nyl = ny ÷ nl

    L = CUDA.CuArray{Int}(undef, nxl, nyl)
    L_host = zeros(Int, nxl, nyl)

    mfl = CUDA.CuArray{Float64}(undef, nxl, nyl)

    dike_a = zeros(ndikes_all)
    dike_b = zeros(ndikes_all)
    dike_x = zeros(ndikes_all)
    dike_y = zeros(ndikes_all)
    dike_t = zeros(ndikes_all)

    h = fopen("dikes.bin", "rb")

    fread!(h, dike_a)
    fread!(h, dike_b)
    fread!(h, dike_x)
    fread!(h, dike_y)
    fread!(h, dike_t)

    fclose(h)

    fid = H5Fopen("particles.h5", H5F_ACC_RDONLY, H5P_DEFAULT)
    read_h5(fid, px, SIZE_1D(npartcl, Float64), staging)
    read_h5(fid, py, SIZE_1D(npartcl, Float64), staging)

    read_h5(fid, px_dikes, SIZE_1D(np_dikes, Float64), staging)
    read_h5(fid, py_dikes, SIZE_1D(np_dikes, Float64), staging)
    H5Fclose(fid)

    fid = H5Fopen("markers.h5", H5F_ACC_RDONLY, H5P_DEFAULT)
    gid = H5Gopen(fid, "0", H5P_DEFAULT)
    read_h5(gid, mx, SIZE_1D(max_nmarker, Float64), staging)
    read_h5(gid, my, SIZE_1D(max_nmarker, Float64), staging)
    read_h5(gid, mT, SIZE_1D(max_nmarker, Float64), staging)
    H5Gclose(gid)
    H5Fclose(fid)

    filename = "grid.$(string(0, pad=NDIGITS)).h5"
    fid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT)
    read_h5(fid, T, SIZE_2D(nx, ny, Float64), staging)
    read_h5(fid, C, SIZE_2D(nx, ny, Float64), staging)
    H5Fclose(fid)

    tm_all = tic()

    # init
    printf("%s initialization              \xb3 ", bar1)
    tm = tic()
    pic_amount_tmp = pic_amount
    pic_amount = 1.0
    blockSize1D = 768
    gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
    g2p<<<gridSize1D, blockSize1D>>>(ALL_ARGS)
    gridSize1D = (max_npartcl - npartcl + blockSize1D - 1) ÷ blockSize1D
    init_particles_T<<<gridSize1D, blockSize1D>>>(pT + npartcl, T_magma, max_npartcl - npartcl)
    init_particles_Ph<<<gridSize1D, blockSize1D>>>(pPh + npartcl, 1, max_npartcl - npartcl)
    gridSize1D = (max_nmarker - nmarker + blockSize1D - 1) ÷ blockSize1D
    init_particles_T<<<gridSize1D, blockSize1D>>>(mT + nmarker, T_magma, max_nmarker - nmarker)
    CUCHECK(cudaDeviceSynchronize())
    pic_amount = pic_amount_tmp
    toc(tm)

    idike = 0
    iSample = 0
    eruptionSteps = []

    # action
    for it in 1:nt
        printf("%s it = %d\n", bar1, it)

        is_eruption = false
        is_intrusion = ndikes[it - 1] > 0

        if it % nerupt == 0
            printf("%s checking melt fraction   \xb3 ", bar2)
            tm = tic()

            blockSizel = (16, 32)
            gridSizel = ((nxl + blockSizel[1] - 1) ÷ blockSizel[1], (nyl + blockSizel[2] - 1) ÷ blockSizel[2])
            average<<<gridSizel, blockSizel>>>(mfl, T, C, nl, nx, ny)
            CUCHECK(cudaDeviceSynchronize())
            ccl(mfl, L, tsh, nxl, nyl)
            cudaMemcpy(L_host, L, SIZE_2D(nxl, nyl, Int), cudaMemcpyDeviceToHost)

            volumes = Dict{Int, Int}()

            for iy in 1:nyl
                if iy * dy * nl < Ly_eruption
                    continue
                end
                for ix in 1:nxl
                    if L_host[(iy - 1) * nxl + ix] >= 0
                        volumes[L_host[(iy - 1) * nxl + ix]] += 1
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
            toc(tm)

            dxl = dx * nl
            dyl = dy * nl

            if maxVol * dxl * dyl >= critVol[iSample]
                printf("%s erupting %07d cells   \xb3 ", bar2, maxVol)
                tm = tic()
                cell_idx = CUDA.CuArray{Int}(undef, maxVol)
                cell_idx_host = zeros(Int, maxVol)

                next_idx = 1
                for idx in 1:nxl * nyl
                    if L_host[idx] == maxIdx
                        cell_idx_host[next_idx] = idx
                        next_idx += 1
                    end
                end

                cudaMemcpy(cell_idx, cell_idx_host, SIZE_1D(maxVol, Int), cudaMemcpyHostToDevice)

                blockSize1D = 512
                gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
                advect_particles_eruption<<<gridSize1D, blockSize1D>>>(px, py, cell_idx, gamma, dxl, dyl, npartcl, maxVol, nxl, nyl)
                CUCHECK(cudaDeviceSynchronize())
                gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D
                advect_particles_eruption<<<gridSize1D, blockSize1D>>>(mx, my, cell_idx, gamma, dxl, dyl, nmarker, maxVol, nxl, nyl)
                CUCHECK(cudaDeviceSynchronize())

                cudaFree(cell_idx)
                free(cell_idx_host)

                iSample += 1

                is_eruption = true
                push!(eruptionSteps, it)

                toc(tm)
            end
        end

        if is_intrusion
            printf("%s inserting %02d dikes       \xb3 ", bar2, ndikes[it - 1])
            tm = tic()

            for i in 1:ndikes[it - 1]
                blockSize1D = 512
                # advect particles
                gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
                advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(px, py, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
                                                                        ndikes[it - 1], npartcl)
                dike_start = particle_edges[idike]
                dike_end = particle_edges[idike + 1]
                np_dike = dike_end - dike_start

                if npartcl + np_dike > max_npartcl
                    fprintf(stderr, "ERROR: number of particles exceeds maximum value, increase capacity\n")
                    exit(EXIT_FAILURE)
                end

                cudaMemcpy(px[npartcl+1:npartcl+np_dike], px_dikes[dike_start:dike_end-1], np_dike * sizeof(Float64), cudaMemcpyDeviceToDevice)
                cudaMemcpy(py[npartcl+1:npartcl+np_dike], py_dikes[dike_start:dike_end-1], np_dike * sizeof(Float64), cudaMemcpyDeviceToDevice)
                npartcl += np_dike
                # advect markers
                gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D
                advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(mx, my, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
                                                                        ndikes[it - 1], nmarker)
                CUCHECK(cudaDeviceSynchronize())
                nmarker += marker_edges[idike + 1] - marker_edges[idike]
                idike += 1
            end
            toc(tm)
        end

        if is_eruption || is_intrusion
            printf("%s p2g interpolation        \xb3 ", bar2)
            tm = tic()

            cudaMemset(T, 0, SIZE_2D(nx, ny, Float64))
            cudaMemset(C, 0, SIZE_2D(nx, ny, Float64))
            cudaMemset(wts, 0, SIZE_2D(nx, ny, Float64))

            blockSize1D = 768
            gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
            p2g_project<<<gridSize1D, blockSize1D>>>(ALL_ARGS)
            CUCHECK(cudaDeviceSynchronize())
            p2g_weight<<<gridSize, blockSize>>>(ALL_ARGS)
            CUCHECK(cudaDeviceSynchronize())
            toc(tm)

            printf("%s particle injection ", bar2)
            tm = tic()
            blockSize1D = 512
            gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
            cudaMemset(pcnt, 0, SIZE_2D(nx, ny, Int))
            count_particles<<<gridSize1D, blockSize1D>>>(pcnt, px, py, dx, dy, nx, ny, npartcl)
            CUCHECK(cudaDeviceSynchronize())

            cudaMemcpy(npartcl_d, npartcl, sizeof(Int), cudaMemcpyHostToDevice)
            min_pcount = 2
            inject_particles<<<gridSize, blockSize>>>(px, py, pT, pPh, npartcl_d, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl)
            CUCHECK(cudaDeviceSynchronize())
            new_npartcl = npartcl
            cudaMemcpy(new_npartcl, npartcl_d, sizeof(Int), cudaMemcpyDeviceToHost)
            if new_npartcl > max_npartcl
                fprintf(stderr, "ERROR: number of particles exceeds maximum value, increase capacity\n")
                exit(EXIT_FAILURE)
            end
            if new_npartcl > npartcl
                printf("(%03d) \xb3 ", new_npartcl - npartcl)
                npartcl = new_npartcl
            else
                printf("(000) \xb3 ", bar2)
            end
            toc(tm)
        end

        printf("%s solving heat diffusion   \xb3 ", bar2)
        tm = tic()
        CUCHECK(cudaMemcpy(T_old, T, SIZE_2D(nx, ny, Float64), cudaMemcpyDeviceToDevice))
        for isub in 1:nsub
            update_T<<<gridSize, blockSize>>>(ALL_ARGS)
            CUCHECK(cudaDeviceSynchronize())
        end
        toc(tm)

        printf("%s g2p interpolation        \xb3 ", bar2)
        tm = tic()
        # particles g2p
        blockSize1D = 1024
        gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D
        g2p<<<gridSize1D, blockSize1D>>>(ALL_ARGS)
        # markers g2p
        gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D
        pic_amount_tmp = pic_amount
        pic_amount = 1.0
        g2p<<<gridSize1D, blockSize1D>>>(T, T_old, C, wts, mx, my, mT, NULL, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, nmarker,
                                            nmarker0)
        CUCHECK(cudaDeviceSynchronize())
        pic_amount = pic_amount_tmp
        toc(tm)

        if it % nout == 0 || is_eruption
            tm = tic()
            printf("%s writing results to disk  \xb3 ", bar2)
            filename = "grid.$(string(it, pad=NDIGITS)).h5"
            dims = [ny, nx]
            chunks = [ny, nx]
            fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
            write_h5d(fid, T, SIZE_2D(nx, ny, Float64), dims, chunks, staging)
            write_h5d(fid, C, SIZE_2D(nx, ny, Float64), dims, chunks, staging)
            if is_eruption
                dims = [nyl, nxl]
                chunks = [nyl, nxl]
                write_h5i(fid, L, SIZE_2D(nxl, nyl, Int), dims, chunks, staging)
            end
            H5Fclose(fid)
            toc(tm)
        end

          tm = tic()
    println("$bar2 writing markers to disk  \xb3 ")
    fid = H5Fopen("markers.h5", H5F_ACC_RDWR, H5P_DEFAULT)
    dims1d = [nmarker]
    chunks1d = [nmarker]
    _gname = string(it)
    gid = H5Gcreate(fid, _gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
    write_h5d(gid, mx, SIZE_1D(nmarker, Float64), dims1d, chunks1d, staging)
    write_h5d(gid, my, SIZE_1D(nmarker, Float64), dims1d, chunks1d, staging)
    write_h5d(gid, mT, SIZE_1D(nmarker, Float64), dims1d, chunks1d, staging)
    H5Gclose(gid)
    H5Fclose(fid)
    toc(tm)
end

println("Total time: ")
toc(tm_all)

h = fopen("eruptions.bin", "wb")
fwrite(h, iSample, sizeof(Int32), 1)
fwrite(h, eruptionSteps, sizeof(Int32), iSample)
fclose(h)

cudaFree(T)
cudaFree(T_old)
cudaFree(C)
cudaFree(px)
cudaFree(py)
cudaFree(pT)
cudaFree(mx)
cudaFree(my)
cudaFree(mT)
cudaFree(mfl)
cudaFree(L)
cudaFreeHost(staging)
cudaFree(npartcl_d)

free(critVol)
free(L_host)
free(dike_a)
free(dike_b)
free(dike_x)
free(dike_y)
free(dike_t)
free(particle_edges)
free(marker_edges)

return 1

