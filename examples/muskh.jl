"""
file to test d2dm preassure model
"""

using Revise
using Dykes2DModel
using CUDA
using Plots
using Random
using Statistics

using Interpolations

include("StressFieldLoader.jl")
using .StressFieldLoader

#using CoordRefSystems

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return [X, Y]
end

# 2D to 1D coordinates
function idc(ix, iy, nx)
    return ((iy) * nx + ix + 1)
end

function log_println(str, level = 1)
    println(("\t"^(level-1)) * ("└─ ") * str)
end

function set_init_sigma(S, g, rhp_r, depth::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + rhp_r * g * (depth - (depth) * (Float32(iy) / Float32(ny)))
    return
end

function set_init_sigma_cald_yy(S, g, rhp_r, dx, Lx::Float32, Ly::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    a = Lx / 3.0
    x::Float32 = ix * dx - Lx / 2
    y::Float32 = Ly - ((Ly) * (Float32(iy + 1) / Float32(ny)))
    p_0 = 5.0
    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + (-p_0 / pi * (atan((x + a) / y) - atan((x - a) / y)))
    return
end

function set_init_sigma_cald_xx(S, g, rhp_r, dx, Lx::Float32, Ly::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    a = Lx / 3.0
    x::Float32 = ix * dx - Lx / 2
    y::Float32 = Ly - ((Ly) * (Float32(iy + 1) / Float32(ny)))
    p_0 = 5.0
    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + (-p_0 / pi * ((atan((x + a) / y) - atan((x - a) / y)) + (y * (x + a) / (y^2 + (x + a)^2)) - (y * (x - a)) / (y^2 + (x - a)^2)))
    return
end

function set_init_sigma_cald_xy(S, g, rhp_r, dx, Lx::Float32, Ly::Float32, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    a = Lx / 3.0
    x::Float32 = ix * dx - Lx / 2
    y::Float32 = Ly - ((Ly) * (Float32(iy + 1) / Float32(ny)))
    p_0 = 5.0
    S[idc(ix, iy, nx)] = S[idc(ix, iy, nx)] + (p_0 / pi * log((y^2 + (x + a)^2) / (y^2 + (x - a)^2)))
    return
end

function init_stress_fields(file_paths::Dict{Symbol, String}, grid_resolution::Int=1)
    
    log_println("Loading stress fields from HDF5 files...", 1)
    
    # Initialize storage
    stress_data = Dict()
    coordinates = Dict()
    
    # Load each stress component
    for (component, file_path) in file_paths
        log_println("Loading $component from $(basename(file_path))", 2)
        
        h5open(file_path, "r") do file
            # Read data
            data = read(file["/$component"])
            x_coords = read(file["/x"])
            z_coords = read(file["/z"])
            
            # Store
            stress_data[component] = data
            coordinates[component] = (x_coords, z_coords)
        end
    end
    
    # Use consistent coordinates (take from Sxx as reference)
    x_ref = coordinates[:Sxx][1]
    z_ref = coordinates[:Sxx][2]
    
    # Create full ranges (consistent for all components)
    x_range = range(minimum(x_ref), maximum(x_ref), length=length(x_ref))
    z_range = range(minimum(z_ref), maximum(z_ref), length=length(z_ref))
    
    # Downsample for plotting and GPU memory
    x_plot = collect(x_range[1:grid_resolution:end])
    z_plot = collect(z_range[1:grid_resolution:end])
    nx = length(x_plot)
    nz = length(z_plot)
    
    log_println("Grid size: $nx x $nz (downsampled by factor $grid_resolution)", 2)
    log_println("Original size: $(length(x_ref)) x $(length(z_ref))", 2)
    
    # Initialize interpolated matrices (x varies first, z second - column-major for GPU)
    Sxx_matrix = Matrix{Float64}(undef, nx, nz)
    Szz_matrix = Matrix{Float64}(undef, nx, nz)
    Sxz_matrix = Matrix{Float64}(undef, nx, nz)
    
    # Interpolate each component
    for (component, data) in stress_data
        log_println("Interpolating $component...", 2)
        
        x_coords, z_coords = coordinates[component]
        
        # Create interpolation object (assuming data is [z, x] as per comment)
        itp = interpolate(data, BSpline(Cubic(Line(OnGrid()))))
        itp = Interpolations.scale(itp, z_coords, x_coords)
        
        # Interpolate onto regular erid (x first for column-major order)
        if component == :Sxx
            for (ix, x) in enumerate(x_plot)
                for (iz, z) in enumerate(z_plot)
                    Sxx_matrix[ix, iz] = itp(z, x)
                end
            end
        elseif component == :Szz
            for (ix, x) in enumerate(x_plot)
                for (iz, z) in enumerate(z_plot)
                    Szz_matrix[ix, iz] = itp(z, x)
                end
            end
        elseif component == :Sxz
            for (ix, x) in enumerate(x_plot)
                for (iz, z) in enumerate(z_plot)
                    Sxz_matrix[ix, iz] = itp(z, x)
                end
            end
        end
    end
    
    # Create GPU arrays (flattened for kernel access)
    log_println("Copying to GPU...", 2)
    Sxx_gpu = CuArray(vec(Sxx_matrix))
    Szz_gpu = CuArray(vec(Szz_matrix))
    Sxz_gpu = CuArray(vec(Sxz_matrix))
    
    log_println("GPU arrays allocated: $(sizeof(Sxx_gpu) / 1024^2) MB each", 2)
    
    return (Sxx_gpu=Sxx_gpu, Szz_gpu=Szz_gpu, Sxz_gpu=Sxz_gpu,
            x_range=x_plot, z_range=z_plot, nx=nx, nz=nz,
            Sxx_plot=Sxx_matrix, Szz_plot=Szz_matrix, Sxz_plot=Sxz_matrix)
end


"""
    init_stress_fields_simple(nx::Int=400, nz::Int=120)

Initialize stress fields with analytical solution (for testing when HDF5 files are unavailable).
"""
function init_stress_fields_simple(nx::Int=400, nz::Int=120)
    log_println("Using simple analytical initialization (no HDF5 files)", 1)
    
    # Create grid
    Lx = 20000.0
    Lz = 6000.0
    x_range = range(0, Lx, nx)
    z_range = range(0, Lz, nz)
    
    # Initialize with lithostatic pressure
    g = 9.8
    rhp_r = 0.02650  # density contrast
    Sxx_matrix = zeros(nx, nz)
    Szz_matrix = zeros(nx, nz)
    Sxz_matrix = zeros(nx, nz)
    
    for (ix, x) in enumerate(x_range)
        for (iz, z) in enumerate(z_range)
            # Lithostatic stress
            pressure = rhp_r * g * (Lz - z)
            Sxx_matrix[ix, iz] = pressure
            Szz_matrix[ix, iz] = pressure
            Sxz_matrix[ix, iz] = 0.0
        end
    end
    
    # Add perturbation (optional - for testing)
    a = Lx / 3.0
    p_0 = 5.0e6
    for (ix, x) in enumerate(x_range)
        for (iz, z) in enumerate(z_range)
            y = Lz - z
            x_centered = x - Lx/2
            
            # Simple pressure perturbation
            perturbation = -p_0/π * (atan((x_centered + a)/y) - atan((x_centered - a)/y))
            Sxx_matrix[ix, iz] += perturbation * 0.1  # Scale for testing
            Szz_matrix[ix, iz] += perturbation * 0.1
        end
    end
    
    # Copy to GPU
    Sxx_gpu = CuArray(vec(Sxx_matrix))
    Szz_gpu = CuArray(vec(Szz_matrix))
    Sxz_gpu = CuArray(vec(Sxz_matrix))
    
    return (Sxx_gpu=Sxx_gpu, Szz_gpu=Szz_gpu, Sxz_gpu=Sxz_gpu,
            x_range=x_range, z_range=z_range, nx=nx, nz=nz,
            Sxx_plot=Sxx_matrix, Szz_plot=Szz_matrix, Sxz_plot=Sxz_matrix)
end

function visualize_propagation_path_simple(propagation_x, propagation_y, xpoints, ypoints, 
                                           initial_x=3000.0, initial_y=2000.0,
                                           xlim=(0,20000), ylim=(0,6000),
                                           bottom_threshold=1001.0)
    # Detect segments based on bottom starts
    segments = []
    current_segment_x = Float64[]
    current_segment_y = Float64[]
    
    if length(propagation_x) > 0
        for i in 1:length(propagation_x)
            if i == 1
                current_segment_x = [propagation_x[i]]
                current_segment_y = [propagation_y[i]]
            else
                # Check if this point starts a new dyke from bottom
                if propagation_y[i] < bottom_threshold && i > 1
                    if length(current_segment_x) > 0
                        push!(segments, (copy(current_segment_x), copy(current_segment_y)))
                    end
                    current_segment_x = [propagation_x[i]]
                    current_segment_y = [propagation_y[i]]
                else
                    push!(current_segment_x, propagation_x[i])
                    push!(current_segment_y, propagation_y[i])
                end
            end
        end
        if length(current_segment_x) > 0
            push!(segments, (current_segment_x, current_segment_y))
        end
    end
    
    # Create plot for propagation path
    figure = Plots.plot(title="Dyke Propagation Path",
                  xlabel="X (m)",
                  ylabel="Z (m)",
                  xlim=xlim,
                  ylim=ylim,
                  aspect_ratio=:equal,
                  legend=:topright,
                  framestyle=:box,
                  grid=true,
                  gridlinewidth=0.5,
                  linewidth=1.5,
                  markersize=3,
                  titlefontsize=10,
                  guidefontsize=9,
                  legendfontsize=8,
                  tickfontsize=8,
                  bg=:white,
                  fg=:black)
    
    # Use same color for all dykes
    uniform_color = :dodgerblue
    
    # Plot each segment
    for (idx, (seg_x, seg_y)) in enumerate(segments)
        # Plot points for this segment
        Plots.scatter!(figure, seg_x, seg_y, 
                markersize=3,
                markercolor=uniform_color,
                markerstrokewidth=0,
                label=idx == 1 ? "Dyke points" : false,
                alpha=0.6)
        
        # Connect points within segment
        if length(seg_x) > 1
            Plots.plot!(figure, seg_x, seg_y,
                  linecolor=uniform_color,
                  linewidth=1.5,
                  linestyle=:solid,
                  label=idx == 1 ? "Dyke path" : false,
                  alpha=0.8)
        end
        
        # Add enumeration numbers
        if length(seg_x) > 0
            annotate!(figure, seg_x[1], seg_y[1], 
                     text(" $idx", :black, :left, 8))
            
            if length(seg_x) > 1
                annotate!(figure, seg_x[end], seg_y[end], 
                         text(" $idx", :black, :right, 8))
            end
        end
    end
    
    # Mark initial dyke start point
    Plots.scatter!(figure, [initial_x], [initial_y], 
             markersize=8,
             markercolor=:red,
             marker=:star,
             label="Initial dyke start",
             markerstrokewidth=0.5,
             markerstrokecolor=:black)
    
    # Mark end points
    if !isempty(segments)
        end_x = [seg_x[end] for (seg_x, _) in segments]
        end_y = [seg_y[end] for (_, seg_y) in segments]
        scatter!(figure, end_x, end_y,
                 markersize=5,
                 markercolor=:purple,
                 marker=:diamond,
                 label="End points",
                 markerstrokewidth=0.5,
                 markerstrokecolor=:black,
                 alpha=0.8)
    end
    
    # Plot local tip points
	#=
    if !isempty(xpoints) && !isempty(ypoints)
        Plots.scatter!(figure, xpoints, ypoints,
                 markersize=1.5,
                 markercolor=:gray,
                 marker=:circle,
                 label="Local tip points",
                 alpha=0.3,
                 markerstrokewidth=0)
    end
    =#
    
    # Print statistics

    total_length = 0.0
    for (idx, (seg_x, seg_y)) in enumerate(segments)
        if length(seg_x) > 1
            seg_length = sum(sqrt.(diff(seg_x).^2 + diff(seg_y).^2))
            total_length += seg_length
            
            dx = seg_x[end] - seg_x[1]
            dy = seg_y[end] - seg_y[1]
            direction = atan(dy, dx) * 180 / π

           #= 
            println("\n" * "-"^50)
            println("DYKE $idx:")
            println("  ├─ Points: $(length(seg_x))")
            println("  ├─ Length: $(round(seg_length, digits=2)) m")
            println("  ├─ Start: ($(round(seg_x[1], digits=2)), $(round(seg_y[1], digits=2)))")
            println("  └─ End:   ($(round(seg_x[end], digits=2)), $(round(seg_y[end], digits=2)))")
            println("  └─ Direction: $(round(direction, digits=1))°")
            println("  └─ X disp: $(round(seg_x[end] - seg_x[1], digits=2)) m")
            println("     └─ Z disp: $(round(seg_y[end] - seg_y[1], digits=2)) m")
		    =#
            
        elseif length(seg_x) == 1
#=
            println("\n" * "-"^50)
            println("DYKE $idx:")
            println("  └─ Single point at ($(round(seg_x[1], digits=2)), $(round(seg_y[1], digits=2)))")
=#
        end
    end
    
    #=
    println("\n" * "="^70)
    println("TOTAL PROPAGATION LENGTH: $(round(total_length, digits=2)) m")
    println("="^70)
	=#
    
    return figure
end

function visualize_local_tip_points(xpoints, ypoints, x_range, z_range, Sxx_plot, dyke_position_x, dyke_position_y)
    """
    Visualize local tip points around the last dyke.
    """
    # Calculate appropriate zoom limits around the dyke position
    zoom_range = 500.0  # 500m zoom around the dyke
    
    x_min = max(minimum(x_range), dyke_position_x - zoom_range)
    x_max = min(maximum(x_range), dyke_position_x + zoom_range)
    z_min = max(minimum(z_range), dyke_position_y - zoom_range)
    z_max = min(maximum(z_range), dyke_position_y + zoom_range)
    
    # Create zoomed heatmap around the dyke
    p_tip = Plots.heatmap(x_range, z_range, Sxx_plot',
                    title="Local Stress Field at Dyke Tip (Step $(length(xpoints)))",
                    xlabel="X (m)", ylabel="Z (m)",
                    clim=(-2e8, 1e8), c=:viridis,
                    xlim=(x_min, x_max), ylim=(z_min, z_max),
                    aspect_ratio=:equal, framestyle=:box,
                    grid=true, gridlinewidth=0.5,
                    titlefontsize=10, guidefontsize=9)
    
    # Plot local tip points
    if !isempty(xpoints) && !isempty(ypoints)
        Plots.scatter!(p_tip, xpoints, ypoints,
                 markersize=4,
                 markercolor=:red,
                 marker=:circle,
                 label="Local tip points ($(length(xpoints)) points)",
                 alpha=0.7,
                 markerstrokewidth=0.5,
                 markerstrokecolor=:black)
        
        # Mark the dyke center
        Plots.scatter!(p_tip, [dyke_position_x], [dyke_position_y],
                 markersize=10,
                 markercolor=:yellow,
                 marker=:star,
                 label="Dyke center",
                 markerstrokewidth=1,
                 markerstrokecolor=:black)
        
        # Connect tip points in order (if they represent a shape)
        if length(xpoints) > 2
            # Close the polygon by connecting back to first point
            x_closed = vcat(xpoints, xpoints[1])
            y_closed = vcat(ypoints, ypoints[1])
            Plots.plot!(p_tip, x_closed, y_closed,
                  linecolor=:blue,
                  linewidth=1.5,
                  linestyle=:dash,
                  label="Tip boundary",
                  alpha=0.6)
        end
        
        # Add statistics
        println("\n=== LOCAL TIP POINTS STATISTICS ===")
        println("Dyke position: ($dyke_position_x, $dyke_position_y)")
        println("Number of tip points: $(length(xpoints))")
        println("Tip span: X [$(round(minimum(xpoints), digits=2)), $(round(maximum(xpoints), digits=2))]")
        println("Tip span: Z [$(round(minimum(ypoints), digits=2)), $(round(maximum(ypoints), digits=2))]")
        
        # Calculate tip shape metrics
        if length(xpoints) > 2
            # Calculate approximate area using shoelace formula
            area = 0.5 * abs(sum(xpoints[i] * (ypoints[i+1] - ypoints[i-1]) for i in 2:length(xpoints)-1) +
                            xpoints[1] * (ypoints[2] - ypoints[end]) +
                            xpoints[end] * (ypoints[1] - ypoints[end-1]))
            println("Approximate tip area: $(round(area, digits=2)) m²")
            
            # Calculate centroid
            cx = sum(xpoints) / length(xpoints)
            cy = sum(ypoints) / length(ypoints)
            println("Tip centroid: ($(round(cx, digits=2)), $(round(cy, digits=2)))")
        end
    end
    
    return p_tip
end

function my_percentile(data, p)
    """
    Calculate percentile of an array.
    
    # Arguments
    - `data`: Array of values
    - `p`: Percentile (0-100)
    """
    if isempty(data)
        return NaN
    end
    sorted_data = sort(data)
    n = length(sorted_data)
    idx = (p / 100) * (n - 1) + 1
    idx_floor = floor(Int, idx)
    idx_ceil = ceil(Int, idx)
    
    if idx_floor == idx_ceil
        return sorted_data[idx_floor]
    else
        # Linear interpolation
        return sorted_data[idx_floor] + (sorted_data[idx_ceil] - sorted_data[idx_floor]) * (idx - idx_floor)
    end
end

function get_stress_statistics(Sxx, Szz, Sxz)
    """
    Print statistics for stress fields to help choose limits.
    """
    println("\n=== STRESS FIELD STATISTICS ===")
    
    # Sxx statistics
    Sxx_flat = Sxx[isfinite.(Sxx)]
    println("Sxx:")
    println("  Min: $(round(minimum(Sxx_flat), digits=2)) Pa")
    println("  Max: $(round(maximum(Sxx_flat), digits=2)) Pa")
    println("  Mean: $(round(mean(Sxx_flat), digits=2)) Pa")
    println("  Std: $(round(std(Sxx_flat), digits=2)) Pa")
    println("  Median: $(round(median(Sxx_flat), digits=2)) Pa")
    
    # Calculate percentiles using my_percentile function
    println("  Percentile 1%: $(round(my_percentile(Sxx_flat, 1), digits=2)) Pa")
    println("  Percentile 99%: $(round(my_percentile(Sxx_flat, 99), digits=2)) Pa")
    
    # Szz statistics
    Szz_flat = Szz[isfinite.(Szz)]
    println("\nSzz:")
    println("  Min: $(round(minimum(Szz_flat), digits=2)) Pa")
    println("  Max: $(round(maximum(Szz_flat), digits=2)) Pa")
    println("  Mean: $(round(mean(Szz_flat), digits=2)) Pa")
    println("  Std: $(round(std(Szz_flat), digits=2)) Pa")
    println("  Median: $(round(median(Szz_flat), digits=2)) Pa")
    println("  Percentile 1%: $(round(my_percentile(Szz_flat, 1), digits=2)) Pa")
    println("  Percentile 99%: $(round(my_percentile(Szz_flat, 99), digits=2)) Pa")
    
    # Sxz statistics
    Sxz_flat = Sxz[isfinite.(Sxz)]
    println("\nSxz:")
    println("  Min: $(round(minimum(Sxz_flat), digits=2)) Pa")
    println("  Max: $(round(maximum(Sxz_flat), digits=2)) Pa")
    println("  Mean: $(round(mean(Sxz_flat), digits=2)) Pa")
    println("  Std: $(round(std(Sxz_flat), digits=2)) Pa")
    println("  Median: $(round(median(Sxz_flat), digits=2)) Pa")
    println("  Percentile 1%: $(round(my_percentile(Sxz_flat, 1), digits=2)) Pa")
    println("  Percentile 99%: $(round(my_percentile(Sxz_flat, 99), digits=2)) Pa")
    
    return nothing
end

function calculate_climits(data, percentile_val=99.0)
    """
    Calculate appropriate color limits for stress plots.
    Uses percentiles to handle outliers.
    
    # Arguments
    - `data`: 2D array of stress values
    - `percentile_val`: Percentile to use for limits (default 99%)
    
    # Returns
    - `(cmin, cmax)`: Color limits
    """
    # Flatten the data to ignore NaNs
    data_flat = data[isfinite.(data)]
    
    if isempty(data_flat)
        return (-1e8, 1e8)
    end
    
    # Calculate percentiles
    p_low = 100 - percentile_val
    p_high = percentile_val
    
    cmin = my_percentile(data_flat, p_low)
    cmax = my_percentile(data_flat, p_high)
    
    # Make symmetric if stresses are roughly symmetric around zero
    if abs(cmin) > abs(cmax) * 0.8 && abs(cmax) > abs(cmin) * 0.8
        # Symmetric limits
        max_abs = max(abs(cmin), abs(cmax))
        cmin = -max_abs
        cmax = max_abs
    end
    
    return (cmin, cmax)
end

"""Convert stress from Pa to MPa (1 MPa = 1e6 Pa)"""
function convert_stress_to_MPa(stress_pa)
    return stress_pa / 1e6
end

"""Convert distance from meters to kilometers"""
function convert_distance_to_km(distance_m)
    return distance_m / 1000.0
end

"""Main funciton to test dykes propagation based on Muskelishvilli solution"""
function d2dm_pres_test()
	log_println("Start of $(stacktrace()[1].func)")
    Random.seed!(1234)

    # Number of modeled dykes
    num_of_dykes = 8
    
    # Define domain boundaries
    X_left_lim, X_right_lim = 0.0, 20000.0
    Z_left_lim, Z_right_lim = 0.0, 6000.0
    z_limit = Z_right_lim - 700
    
    # Initialize stress fields
    log_println("Initializing stress fields...")
    data = StressFieldLoader.get_all_stress_data(resolution=5, on_gpu=true)
    
    # Extract data
    Sxx_gpu = reverse(data["Sxx_gpu"], dims=1)
    Szz_gpu = reverse(data["Szz_gpu"], dims=1)
    Sxz_gpu = data["Sxz_gpu"]

    x_range = data["x_range"]
    z_range = data["z_range"]
    nx = data["nx"]
    nz = data["nz"]
    Sxx_plot = data["Sxx_plot"]
    Szz_plot = data["Szz_plot"]
    Sxz_plot = data["Sxz_plot"]
    
    log_println("Diagnostics")
    log_println("Grid size: $nx x $nz", 2)
    log_println("X range: $(minimum(x_range)) to $(maximum(x_range))", 2)
    log_println("Z range: $(minimum(z_range)) to $(maximum(z_range))", 2)
    
    # Create GPU arrays for coordinates
    x_range_gpu = CuArray(collect(Float64, x_range))
    z_range_gpu = CuArray(collect(Float64, z_range))
    
    # Initialize dyke parameter
    dyke_param = DykeParam(x=3000.0, y=2000.0, phi=π/2, a=500.0, b=20.0, P_in=2e6)
    
    # Storage for dyke propagation path
    xpoints = Float64[]
    ypoints = Float64[]
    l_vecs = zeros(4)
    
    # Grid params
    blockSize = (16, 16)
    gridSize = (Int(ceil((nx + blockSize[1] - 1) / blockSize[1])), 
                Int(ceil((nz + blockSize[2] - 1) / blockSize[2])))
    
    log_println("Grid size for kernel: ($(gridSize[1]), $(gridSize[2]))", 2)
    log_println("Block size: ($(blockSize[1]), $(blockSize[2]))", 2)
    
    log_println("Main loop", 1)

    propagation_x = Float64[]
    propagation_y = Float64[]

    # Storage for tip points of each dyke (to show evolution)
    all_tip_points_x = Vector{Vector{Float64}}()
    all_tip_points_y = Vector{Vector{Float64}}()
    
    for i in 1:num_of_dykes
         begin
            @cuda blocks=gridSize[1], gridSize[2] threads=blockSize[1], blockSize[2] insert_dyke_gpu!(Sxx_gpu, Szz_gpu, Sxz_gpu, x_range_gpu, z_range_gpu, nx, nz,
                    Float64(dyke_param.a), Float64(dyke_param.b), Float64(dyke_param.P_in), 
                    Float64(dyke_param.x), Float64(dyke_param.y), Float64(dyke_param.phi))
            
            synchronize()
            
            copyto!(vec(Sxx_plot), Sxx_gpu)
            copyto!(vec(Szz_plot), Szz_gpu)
            copyto!(vec(Sxz_plot), Sxz_gpu)
            
            next_point_x, next_point_y, xpoints, ypoints, l_vecs = calc_cent_of_next_dyke(dyke_param, Sxx_plot, Szz_plot, Sxz_plot, x_range, z_range, z_limit, Z_right_lim, X_right_lim)

            push!(propagation_x, next_point_x)
            push!(propagation_y, next_point_y)

		    # Store tip points for this step
            push!(all_tip_points_x, copy(xpoints))
            push!(all_tip_points_y, copy(ypoints))

            log_println("Dyke #$i inserted!", 2)
            #log_println("Coordinates - ($next_point_x, $next_point_y) inserted!", 2)

            _, phi_tmp = d2dm_cart_to_polar(l_vecs[3], l_vecs[4])
            dyke_param = DykeParam(x=next_point_x, y=next_point_y, phi=phi_tmp)
        end
    end

	copyto!(vec(Sxx_plot), Sxx_gpu)
	copyto!(vec(Szz_plot), Szz_gpu)
	copyto!(vec(Sxz_plot), Sxz_gpu)

	# Visualize
	log_println("Building graphs", 1)

	# Print statistics to help choose limits
	Sxx_plot_MPa = convert_stress_to_MPa.(Sxx_plot)
	Szz_plot_MPa = convert_stress_to_MPa.(Szz_plot)
	Sxz_plot_MPa = convert_stress_to_MPa.(Sxz_plot)

	# Convert coordinates from meters to kilometers
	x_range_km = convert_distance_to_km.(x_range)
	z_range_km = convert_distance_to_km.(z_range)

	# Print statistics in MPa
    # Calculate appropriate limits in MPa
	# For Sxx and Szz (compressive stresses)
	sxx_min_MPa = minimum(Sxx_plot_MPa)
	sxx_max_MPa = maximum(Sxx_plot_MPa)
	szz_min_MPa = minimum(Szz_plot_MPa)
	szz_max_MPa = maximum(Szz_plot_MPa)

	# For Sxz (shear stress) - symmetric around zero
	sxz_max_abs_MPa = max(abs(minimum(Sxz_plot_MPa)), abs(maximum(Sxz_plot_MPa)))
	sxz_min_MPa = -sxz_max_abs_MPa
	sxz_max_MPa = sxz_max_abs_MPa

	# Create transposed views for plotting (z vs x)
	Sxx_viz_MPa = Sxx_plot_MPa'
	Szz_viz_MPa = Szz_plot_MPa'
	Sxz_viz_MPa = Sxz_plot_MPa'

	x_plot_km = collect(x_range_km)
	z_plot_km = collect(z_range_km)

	x_lim_km = (minimum(x_range_km), maximum(x_range_km))
	z_lim_km = (minimum(z_range_km), maximum(z_range_km))    

	# In the heatmap plotting section, change ylim to be reversed
	p1 = Plots.heatmap(x_plot_km, z_plot_km, Sxx_viz_MPa, 
					 title="Sxx (MPa) - Compressive",
					 xlabel="X (km)", ylabel="Z (km)",
				 xlim=x_lim_km, ylim=z_lim_km,
					 clim=(sxx_min_MPa, sxx_max_MPa),
					 c=:viridis,
					 aspect_ratio=:equal, framestyle=:box,
					 grid=true, gridlinewidth=0.5,
					 titlefontsize=10, guidefontsize=9)

	p2 = Plots.heatmap(x_plot_km, z_plot_km, Szz_viz_MPa, 
					 title="Szz (MPa) - Compressive",
					 xlabel="X (km)", ylabel="Z (km)",
				 xlim=x_lim_km, ylim=z_lim_km,
					 clim=(szz_min_MPa, szz_max_MPa),
					 c=:viridis,
					 aspect_ratio=:equal, framestyle=:box,
					 grid=true, gridlinewidth=0.5,
					 titlefontsize=10, guidefontsize=9)

	p3 = Plots.heatmap(x_plot_km, z_plot_km, Sxz_viz_MPa, 
					 title="Sxz (MPa) - Shear",
					 xlabel="X (km)", ylabel="Z (km)",
				 xlim=x_lim_km, ylim=z_lim_km,
					 clim=(sxz_min_MPa, sxz_max_MPa),
					 c=:balance,
					 aspect_ratio=:equal, framestyle=:box,
					 grid=true, gridlinewidth=0.5,
					 titlefontsize=10, guidefontsize=9)	# Also update the propagation path plot to use km

	propagation_x_km = convert_distance_to_km.(propagation_x)
	propagation_y_km = convert_distance_to_km.(propagation_y)
	xpoints_km = convert_distance_to_km.(xpoints)
	ypoints_km = convert_distance_to_km.(ypoints)
	initial_x_km = 3.0  # 3000m = 3km
	initial_y_km = 2.0  # 2000m = 2km

	# Create propagation path plot with km units
	p4 = visualize_propagation_path_simple(propagation_x_km, propagation_y_km, 
										   xpoints_km, ypoints_km,
										   initial_x_km, initial_y_km,
										   (0, 20), (0, 6),  # xlim, ylim in km
										   1.001)  # bottom_threshold in km
	# Combine all plots - now 5 plots (2x3 layout, but 5 plots so one empty)
	final_plot = Plots.plot(p1, p2, p3, p4,
					 layout=(2,2),
					 size=(1800, 1200),
					 margin=5*Plots.mm,
					 plot_title="Dyke Propagation Simulation Results - Complete Analysis",
					 plot_titlefontsize=14)

	#display(final_plot)

	# Save individual plots if needed
	name_of_figure_file::String = "final.png"
	log_println("Saving graphs")
#	Plots.savefig(final_plot, name_of_figure_file)

	# Print tip points evolution summary
	log_println("Tip points evolution summary")
	for i in 1:length(all_tip_points_x)
		n_points = length(all_tip_points_x[i])
		if n_points > 0
			log_println("Dyke $i: $(n_points) tip points at position ($(propagation_x[i]), $(propagation_y[i]))", 2)
		end
	end

	log_println("End of $(stacktrace()[1].func)!", 1)
	return final_plot
end
