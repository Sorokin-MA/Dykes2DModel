using Dates
include("dykes_init.jl")

using PlotlyJS, CSV, DataFrames
using Plots

function test_main()


end


function d2d_test_write()
    vp = VarParams()#scalar params
    gp = GridParams()#array params

    vp.Lx = 8484
    gp.critVol = [1.0, 2.0]
    #wts = CuArray{Float64,1}()
    gp.wts = CuArray{Float64,1}(undef, 2)
    copyto!(gp.wts, gp.critVol)
    filename_donwload = "tesst_6.h5"

    #filename_donwload = @sprintf("d2d_snapshot_%d_%s.hdf5",vp.it, Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))
    #filename_donwload = @sprintf("d2d_config_%s.hdf5",dates.format(now(), "yyyy_mm_dd_hh_mm_ss"))

    if isfile(filename_donwload)
        rm(filename_donwload)
    end

    fid = h5open(filename_donwload, "w")

    for n in fieldnames(typeof(vp))
        println(getfield(vp, n))
        write(fid, string(n), getfield(vp, n))
    end

    for n in fieldnames(typeof(gp))
        if (getfield(gp, n) isa CuArray)
            d2d_cu_type = eltype(getfield(gp, n))

            nn::Array{d2d_cu_type,1} = Array{d2d_cu_type,1}(undef, size(getfield(gp, n))[1])
            copyto!(nn, getfield(gp, n))

            println(getfield(gp, n))
            write(fid, string(n), nn)
            println("sucess!!")
        else
            println(getfield(gp, n))
            write(fid, string(n), getfield(gp, n))
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

    vp = VarParams()#scalar params
    gp = GridParams()#array params

    fid = h5open(filename_donwload, "r")
    #init_vp = InitVarParams()	#params for generate random

    #init_vp.critVol = global_EruptionVolumesVec
    #init_vp.critVolTime = global_EruptionTimesVec

    for n in fieldnames(typeof(vp))
        setfield!(vp, n, read(fid, string(n)))
        println(getfield(vp, n))
    end

    for n in fieldnames(typeof(gp))
        if (getfield(gp, n) isa CuArray)
            d2d_cu_type = eltype(getfield(gp, n))
            nn::CuArray{d2d_cu_type,1} = CuArray{d2d_cu_type,1}(undef, size(getfield(gp, n))[1])
            #copyto!(nn, read(fid, string(n)))
            nn = read(fid, string(n))
            setfield!(gp, n, nn)
            #copyto!(getfield(gp,n), read(fid, string(n)))
            #copyto!(getfield(gp,n), read(fid, string(n)))
            #write(fid, string(n), nn)

            println("GPU")
            println(getfield(gp, n))
        else
            #read(fid, string(n), getfield(gp,n))


            #d2d_cu_type = eltype(getfield(gp,n))
            #nnn::Array{d2d_cu_type,1} = Array{d2d_cu_type,1}(undef, size(getfield(gp,n))[1]);

            #nnn = read(fid, string(n))
            setfield!(gp, n, read(fid, string(n)))
            println("CPU")
            println(getfield(gp, n))
        end
    end

    println("success?")


    close(fid)
end
