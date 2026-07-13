module StressFieldLoader

using HDF5, Interpolations, CUDA, JLD2, Statistics

export get_stress_fields, get_gpu_arrays, clear_cache, init_stress_fields, get_all_stress_data

# Module-level cache
const _STRESS_CACHE = Dict{String,Any}()

function get_stress_fields(; resolution::Int=5, force_reload::Bool=false)
    cache_key = "res$(resolution)"
    
    if force_reload || !haskey(_STRESS_CACHE, cache_key)
        _STRESS_CACHE[cache_key] = _load_stress_fields(resolution)
    end
    
    return _STRESS_CACHE[cache_key]
end

function _load_stress_fields(resolution::Int)
    # Try binary cache first
    cache_file = "stress_cache_res$(resolution).jld2"
    
    if isfile(cache_file)
        @info "Loading from cache file: $cache_file"
        return load(cache_file)
    end
    
    @info "Loading and interpolating stress fields (this may take a while)..."
    
    # Load HDF5 files
    @info "  Reading Sxx.h5..."
    Sxx_file = h5open("Sxx.h5", "r")
    Sxx = read(Sxx_file["/Sxx"])
    Sxx_x = read(Sxx_file["/x"])
    Sxx_z = read(Sxx_file["/z"])
    close(Sxx_file)
    
    @info "  Reading Szz.h5..."
    Szz_file = h5open("Szz.h5", "r")
    Szz = read(Szz_file["/Szz"])
    Szz_x = read(Szz_file["/x"])
    Szz_z = read(Szz_file["/z"])
    close(Szz_file)
    
    @info "  Reading Sxz.h5..."
    Sxz_file = h5open("Sxz.h5", "r")
    Sxz = read(Sxz_file["/Sxz"])
    Sxz_x = read(Sxz_file["/x"])
    Sxz_z = read(Sxz_file["/z"])
    close(Sxz_file)
    
    # Create ranges (matching your original code)
    x_range = range(0, maximum(Sxx_x) - minimum(Sxx_x), length=length(Sxx_x))
    z_range = range(0, maximum(Sxx_z) - minimum(Sxx_z), length=length(Sxx_z))
    
    # Downsample for plotting and GPU memory
    x_plot = x_range[1:resolution:end]
    z_plot = z_range[1:resolution:end]
    nx = length(x_plot)
    nz = length(z_plot)
    
    @info "Original grid size: $(length(Sxx_x)) x $(length(Sxx_z))"
    @info "Downsampled grid size: $nx x $nz"
    
    # Create interpolation objects (matching your original approach)
    @info "  Creating interpolation objects..."
    itp_xx = interpolate(Sxx, BSpline(Cubic(Line(OnGrid()))))
    itp_xx = scale(itp_xx, z_range, x_range)
    
    itp_zz = interpolate(Szz, BSpline(Cubic(Line(OnGrid()))))
    itp_zz = scale(itp_zz, z_range, x_range)
    
    itp_xz = interpolate(Sxz, BSpline(Cubic(Line(OnGrid()))))
    itp_xz = scale(itp_xz, z_range, x_range)
    
    # Interpolate onto grid (matching your original [z, x] order)
    @info "  Interpolating Sxx..."
    Sxx_plot = [itp_xx(z, x) for z in z_plot, x in x_plot]
    
    @info "  Interpolating Szz..."
    Szz_plot = [itp_zz(z, x) for z in z_plot, x in x_plot]
    
    @info "  Interpolating Sxz..."
    Sxz_plot = [itp_xz(z, x) for z in z_plot, x in x_plot]
    
    # Prepare result
    result = Dict{String,Any}(
        "Sxx_plot" => Sxx_plot,
        "Szz_plot" => Szz_plot,
        "Sxz_plot" => Sxz_plot,
        "x_range" => collect(x_plot),
        "z_range" => collect(z_plot),
        "nx" => nx,
        "nz" => nz
    )
    
    # Save to cache file for next time
    @info "Saving to cache file: $cache_file"
    save(cache_file, result)
    
    return result
end

function get_gpu_arrays(; resolution::Int=5)
    cache = get_stress_fields(resolution=resolution)
    
    # Note: The data is [z, x] orientation, need to transpose for GPU kernel
    # GPU kernel expects [x, z] orientation (x first, then z)
    Sxx_matrix = cache["Sxx_plot"]'  # Transpose to [x, z]
    Szz_matrix = cache["Szz_plot"]'  # Transpose to [x, z]
    Sxz_matrix = cache["Sxz_plot"]'  # Transpose to [x, z]
    
    # Create GPU arrays
    return (
        Sxx_gpu = CuArray(vec(Sxx_matrix)),
        Szz_gpu = CuArray(vec(Szz_matrix)),
        Sxz_gpu = CuArray(vec(Sxz_matrix))
    )
end

function clear_cache()
    empty!(_STRESS_CACHE)
    for file in readdir()
        if startswith(file, "stress_cache_res") && endswith(file, ".jld2")
            rm(file)
            @info "Deleted cache file: $file"
        end
    end
end

function get_all_stress_data(; resolution::Int=5, on_gpu::Bool=true)
    cpu_data = get_stress_fields(resolution=resolution)
    
    if on_gpu
        gpu_data = get_gpu_arrays(resolution=resolution)
        return merge(cpu_data, Dict(
            "Sxx_gpu" => gpu_data.Sxx_gpu,
            "Szz_gpu" => gpu_data.Szz_gpu,
            "Sxz_gpu" => gpu_data.Sxz_gpu
        ))
    else
        return cpu_data
    end
end

# Simple initialization without interpolation (for testing)
function init_stress_fields_simple(; nx::Int=400, nz::Int=120)
    @info "Using simple analytical initialization (no HDF5 files)"
    
    Lx = 20000.0
    Lz = 6000.0
    x_range = range(0, Lx, nx)
    z_range = range(0, Lz, nz)
    
    g = 9.8
    rhp_r = 0.02650
    Sxx_matrix = zeros(nx, nz)
    Szz_matrix = zeros(nx, nz)
    Sxz_matrix = zeros(nx, nz)
    
    for (ix, x) in enumerate(x_range)
        for (iz, z) in enumerate(z_range)
            pressure = rhp_r * g * (Lz - z)
            Sxx_matrix[ix, iz] = pressure
            Szz_matrix[ix, iz] = pressure
            Sxz_matrix[ix, iz] = 0.0
        end
    end
    
    # Add perturbation
    a = Lx / 3.0
    p_0 = 5.0e6
    for (ix, x) in enumerate(x_range)
        for (iz, z) in enumerate(z_range)
            y = Lz - z
            x_centered = x - Lx/2
            perturbation = -p_0/π * (atan((x_centered + a)/y) - atan((x_centered - a)/y))
            Sxx_matrix[ix, iz] += perturbation * 0.1
            Szz_matrix[ix, iz] += perturbation * 0.1
        end
    end
    
    Sxx_gpu = CuArray(vec(Sxx_matrix))
    Szz_gpu = CuArray(vec(Szz_matrix))
    Sxz_gpu = CuArray(vec(Sxz_matrix))
    
    return (Sxx_gpu=Sxx_gpu, Szz_gpu=Szz_gpu, Sxz_gpu=Sxz_gpu,
            x_range=collect(x_range), z_range=collect(z_range), nx=nx, nz=nz,
            Sxx_plot=Sxx_matrix, Szz_plot=Szz_matrix, Sxz_plot=Sxz_matrix)
end

end # module