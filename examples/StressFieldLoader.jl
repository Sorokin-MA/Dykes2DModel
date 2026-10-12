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
    
    # Normalize raw arrays to [x, z] orientation (x along dim 1, z along dim 2).
    # When read in Julia the dataset may come back as [z, x]; fix that here.
    if size(Sxx, 1) == length(Sxx_z) && size(Sxx, 2) == length(Sxx_x)
        Sxx = permutedims(Sxx)
    end
    if size(Szz, 1) == length(Szz_z) && size(Szz, 2) == length(Szz_x)
        Szz = permutedims(Szz)
    end
    if size(Sxz, 1) == length(Sxz_z) && size(Sxz, 2) == length(Sxz_x)
        Sxz = permutedims(Sxz)
    end

    # The stored z axis runs surface -> depth (z[1] is the surface). The model uses
    # z = 0 for the bottom (depth) and z = max for the surface, so flip the z dimension.
    Sxx = reverse(Sxx, dims=2)
    Szz = reverse(Szz, dims=2)
    Sxz = reverse(Sxz, dims=2)
    
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
    
    # Create interpolation objects (raw HDF5 data is [x, z] - x along dim 1, z along dim 2)
    @info "  Creating interpolation objects..."
    itp_xx = interpolate(Sxx, BSpline(Cubic(Line(OnGrid()))))
    itp_xx = scale(itp_xx, x_range, z_range)
    
    itp_zz = interpolate(Szz, BSpline(Cubic(Line(OnGrid()))))
    itp_zz = scale(itp_zz, x_range, z_range)
    
    itp_xz = interpolate(Sxz, BSpline(Cubic(Line(OnGrid()))))
    itp_xz = scale(itp_xz, x_range, z_range)
    
    # Interpolate onto grid in explicit [x, z] order (x = rows dim1, z = cols dim2)
    @info "  Interpolating Sxx..."
    Sxx_plot = Matrix{Float64}(undef, nx, nz)
    for (ix, x) in enumerate(x_plot)
        for (iz, z) in enumerate(z_plot)
            Sxx_plot[ix, iz] = itp_xx(x, z)
        end
    end
    
    @info "  Interpolating Szz..."
    Szz_plot = Matrix{Float64}(undef, nx, nz)
    for (ix, x) in enumerate(x_plot)
        for (iz, z) in enumerate(z_plot)
            Szz_plot[ix, iz] = itp_zz(x, z)
        end
    end
    
    @info "  Interpolating Sxz..."
    Sxz_plot = Matrix{Float64}(undef, nx, nz)
    for (ix, x) in enumerate(x_plot)
        for (iz, z) in enumerate(z_plot)
            Sxz_plot[ix, iz] = itp_xz(x, z)
        end
    end
    
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
    
    # Note: The data is already [x, z] orientation, which is what the GPU kernel expects.
    # GPU kernel expects [x, z] orientation (x first, then z), so no transpose is needed.
    # vec() flattens column-major: element [ix, iz] at index (ix-1) + nx*(iz-1),
    # matching cur_index = ix + nx*(iy-1) in insert_dyke_gpu!.
    Sxx_matrix = cache["Sxx_plot"]
    Szz_matrix = cache["Szz_plot"]
    Sxz_matrix = cache["Sxz_plot"]
    
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