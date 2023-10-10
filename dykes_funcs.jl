#Function that write which device we use
#
include("dykes_init.jl")

function kernel()
    dev = Ref{Cint}()
    CUDA.cudaGetDevice(dev)
    @cuprintln("Running on device $(dev[])")
    return
end

#Function that upload packages mostly
function dykes_init()
    #using Pkg
    #Pkg.status()
    #Pkg.add("HDF5")
    #Pkg.add("CUDA")
    #Pkg.add("JupyterFormatter")
    #using CUDA
    #using Printf
    #using BenchmarkTools
    CUDA.versioninfo()
    collect(devices())

    kernel()

    #using HDF5

end

function print_gpu_properties()

    for (i, device) in enumerate(CUDA.devices())
        println("*** General properties for device $i ***")
        name = CUDA.name(device)
        println("Device name: $name")
        major = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        minor = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        println("Compute capabilities: $major.$minor")
        clock_rate = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
        println("Clock rate: $clock_rate")
        device_overlap = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP)
        print("Device copy overlap: ")
        println(device_overlap > 0 ? "enabled" : "disabled")
        kernel_exec_timeout =
            CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)
        print("Kernel execution timeout: ")
        println(kernel_exec_timeout > 0 ? "enabled" : "disabled")
    end
end

function read_par(par, ipar)
    par_name_2 = par[ipar]
    ipar_2 = ipar + 1
    return par_name_2, ipar_2
end

function idc(ix, iy, nx)
    return (iy) * nx + (ix)
end

function blerp(x1, x2, y1, y2, f11, f12, f21, f22, x, y)
    invDxDy = 1.0 / ((x2 - x1) * (y2 - y1))

    dx1 = x - x1
    dx2 = x2 - x

    dy1 = y - y1
    dy2 = y2 - y

    return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1)
end

function g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)
    ip = blockIdx().x * blockDim().x + threadIdx().x;

    if ip > (npartcl - 1)
        return nothing
    end

    pxi = px[ip]/ dx;
    pyi = px[ip]/ dy;

    ix1 = min(max(convert(Int64, pxi), 0), nx - 2);
    iy1 = min(max(convert(Int64, pyi), 0), ny - 2);
    
    ix2 = ix1 + 1;
    iy2 = iy1 + 1;
    
    x1 = ix1 * dx
    x2 = ix2 * dx;
    y1 = iy1 * dy;
    y2 = iy2 * dy;
    
    T_pic = blerp(x1, x2, y1, y2, T[idc(ix1, iy1, nx)], T[idc(ix1, iy2, nx)], T[idc(ix2, iy1, nx)], T[idc(ix2, iy2, nx)], px[ip], py[ip]);
    T_flip = pT[ip] + T_pic - blerp(x1, x2, y1, y2, T_old[idc(ix1, iy1, nx)], T_old[idc(ix1, iy2, nx)], T_old[idc(ix2, iy1, nx)], T_old[idc(ix2, iy2,nx)], px[ip], py[ip]);
    pT[ip] = T_pic * pic_amount + T_flip * (1.0 - pic_amount);
    
    return nothing
end

function init_particles_T(pT, T_magma, maxpartcl, npartcl)
    ip = blockIdx().x * blockDim().x + threadIdx().x

    if (ip > (maxpartcl - npartcl - 1))
        return
    end

    pT[ip+npartcl] = T_magma
    return
end

#avg += mf_magma(T[idc(ix, iy, nx)]) * vf + mf_rock(T[idc(ix, iy, nx)]) * (1 - vf);
function mf_magma(T)
    t2 = T * T
    t7 = exp(
        0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 -
        0.1866187556e-5 * t2 * T,
    )
    return 0.1e1 / (0.1e1 + t7)
end

#=
double mf_rhyolite(double T) {
  double t2 = T * T;
  double t7 = exp(0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 - 0.1866187556e-5 * t2 * T);
  return 0.1e1 / (0.1e1 + t7);
}
=#

function average(mfl, T, C, nl, nx, ny)
    ixl = blockIdx().x * blockDim().x + threadIdx().x
    iyl = blockIdx().y * blockDim().y + threadIdx().y

    if (ixl > (nx / nl - 1) || iyl > (ny / nl - 1))
        return
    end

    avg = 0.0

    #for (int ix = ixl * nl; ix < (ixl + 1) * nl; ++ix) 
    for ix = (ixl*nl):((ixl+1)*nl)

        if (ix > nx - 1)
            break
        end

        #for (int iy = iyl * nl; iy < (iyl + 1) * nl; ++iy) 
        for iy = (iyl*nl):(iy<(iyl+1)*nl)
            if (iy > ny - 1)
                break
            end
            vf = C[idc(ix, iy, nx)]
            avg +=
                mf_magma(T[idc(ix, iy, nx)]) * vf + mf_rock(T[idc(ix, iy, nx)]) * (1 - vf)
            #=  
            avg += mf_magma(T[idc(ix, iy, nx)]) * vf + mf_rock(T[idc(ix, iy, nx)]) * (1 - vf);
            =#
        end
    end

    avg /= nl * nl
    mfl[iyl*(nx/nl)+ixl] = avg

end


