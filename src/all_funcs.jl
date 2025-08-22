"""
File where all main functions for now for d2dm project
"""

#count max threads to avoid cuda errors
dev_thread::Integer= CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

# These arrays contain data on dyke crystalinity and temperature respectively.
# dykes_crystalinity: A vector of Float64 representing the crystalinity of dykes over a series of measurements.
# dykes_temp: A vector of Float64 representing the temperature of these dykes during the same measurement points.
#NOTE: dykes_temp crystalinity taken uniformly in range of dykes_temp for interpolation purposes
# - Forni F, Degruyter W, Bachmann O, De Astis G, Mollo S. Long-term magmatic evolution reveals the beginning of a new caldera cycle at Campi Flegrei. Sci Adv. 2018 Nov 14;4(11):eaat9401. doi: 10.1126/sciadv.aat9401. PMID: 30788429; PMCID: PMC6371846.
dykes_crystalinity = 1 .- Vector{Float64}([1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0])
dykes_temp = Vector{Float64}([699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385])

A_x = range(699.2355, 1186.2385, 51);
itp = interpolate(dykes_crystalinity, BSpline(Cubic(Line(OnGrid()))))
itp = Interpolations.scale(itp, A_x)
itp = extrapolate(itp, Flat())

cuitp = adapt(CuArray{eltype(dykes_temp)}, itp);


"""
Function to count 1D index based on 2D indexes.

2D indexes start with 0
1D indexes start with 1
It's all due to translation from CUDA code.
Maybe will be fixed in the future.
...
# Arguments
- `ix::Integer`: x coordinate in 2D, starts with 0.
- `iy::Integer`: y coordinate in 2D, starts with 0.
- `nx::Integer`: x dimension size.
...
"""
function idc(ix::Integer, iy::Integer, nx::Integer)
    return iy * nx + ix + 1
end

"""
Reads a parameter from an array and returns the parameter value and the next index.

# Arguments:
- `par_val`: An array containing parameter values.
- `par_index`: The current index in the `par` array.

# Returns:
- A tuple containing the parameter value at the current index and the next index.
"""
function read_par(par_val, par_index::Integer)
    par_val = par_val[par_index]
    par_index_next = par_index + 1
    return par_val, par_index_next
end


"""
blerp is a bilinear interpolation function used to estimate the value of a two-dimensional
function at an arbitrary point within a given rectangle.

The function takes 4 known values of the function (f11, f12, f21, f22) at the four corners
of a rectangle defined by points (x1, y1), (x1, y2), (x2, y1), (x2, y2) and the target point (x, y). It returns
an estimated value based on these known points.

# Arguments
- `x1`, `y1`: Coordinates of the first corner of the rectangle.
- `x2`, `y2`: Coordinates of the opposite corner of the rectangle.
- `f11`, `f12`, `f21`, `f22`: Values of the function at the corners (x1, y1), (x1, y2), (x2, y1), and (x2, y2) respectively.
- `x`, `y`: Coordinates of the point where the interpolated value is needed.

# Returns
- The estimated value of the function at the point (x, y) using bilinear interpolation.

"""
function blerp(x1, x2, y1, y2, f11, f12, f21, f22, x, y)
    invDxDy = 1.0 / ((x2 - x1) * (y2 - y1))

    dx1 = x - x1
    dx2 = x2 - x

    dy1 = y - y1
    dy2 = y2 - y

    return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1)
end


"""
Calculate the melt fraction for rhyolite based on temperature.

This function computes the melt fraction for rhyolite rock sample as a function of its temperature.
NOTE: source unknown

# Arguments
- `T`: Temperature of the rock sample in degrees Celsius.

# Returns
- Melt fraction for rhyolite rock sample.
"""
function mf_rhyolite(T)
    t2 = T * T
    t7 = exp(0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 - 0.1866187556e-5 * t2 * T)
    return 0.1e1 / (0.1e1 + t7)
end

"""
Calculates the derivative of melt fraction (dmf) for rhyolite rock as a function of temperature T.
This function uses a polynomial and exponential fit to approximate the behavior of dmf with temperature.
NOTE: source unknown

# Arguments
- `T`: Temperature in degrees Celsius

# Returns
- The calculated value of the derivative of melt fraction (dmf) for rhyolite rock at the given temperature T.

"""
function dmf_rhyolite(T)
    t1 = T * T
    t9 = exp(0.961026e3 - 0.186618e-5 * t1 * T + t1 * 0.447948e-2 + T * (-0.359050e1))
    t12 = (0.1e1 + t9) * (0.1e1 + t9)
    return 0.559856e-5 / t12 * t9 * (t1 - 0.160022e4 * T + 0.641326e6)
end

"""
Calculate the melt fraction for basalt rocks, based on temperature.
NOTE: source unknown

# Arguments
- `T`: Temperature of the rock sample in degrees Celsius.

# Returns
- Melt fraction for basalt rock sample.
"""
function mf_basalt(T)
    t2 = T * T
    t7 = exp(960 - 3.554 * T + 0.4468e-2 * t2 - 1.907e-06 * t2 * T)
    return 0.1e1 / (0.1e1 + t7)
end

"""
Calculates the derivative of melt fraction (dmf) for basalt rock as a function of temperature T.
NOTE: source unknown

# Arguments
- `T`: Temperature in degrees Celsius

# Returns
- The calculated value of the derivative of melt fraction (dmf) for basalt rock at the given temperature T.

"""
function dmf_basalt(T)
    t1 = T * T
    t11 = exp(0.143636887899999948e3 - 0.2214446257e-6 * t1 * T + t1 * 0.572468110399999928e-3 + T * (-0.494427718499999891e0))
    t14 = (0.1e1 + t11)^0.2e1
    return 0.6643338771e-6 * (t1 - 0.1723434948e4 * T + 0.7442458310e6) * t11 / t14
end


"""
    dmf_magma(T)

Calculate the derivative of melt fraction with respect to temperature for magma.

# Arguments:
- `T`: Temperature in degrees Celsius.

# Returns:
The derivative of melt fraction at temperature T.
"""
function dmf_magma(T)
    return dmf_basalt(T)
end

"""
    mf_magma(T)

Calculate melt fraction with respect to temperature for magma.

# Arguments:
- `T`: Temperature in degrees Celsius.

# Returns:
The melt fraction at temperature T.
"""
function mf_magma(T)
    return mf_basalt(T)
end


"""
    mf_magma(T)

Calculate melt fraction with respect to temperature for host rocks.

# Arguments:
- `T`: Temperature in degrees Celsius.

# Returns:
The melt fraction at temperature T.
"""
function dmf_rock(T)
    return only.(Interpolations.gradient.(Ref(cuitp), T))
end


"""
This function calculates melt fraction of host rocks.

Arguments:
- T: Temperature (in degrees Celsius)

Returns:
- melt fraction value.
"""
function mf_rock(T)
    return cuitp(T)
end


#TODO: fix this export, probably duplicate

function d2dm_mf_rock(T)
    return itp(T)
end

function d2dm_dmf_rock(T)
    #FIXME: bad interpolation?
    return only.(Interpolations.gradient.(Ref(cuitp), T))
end

function d2dm_mf_magma(T)
    return mf_magma(T)
end

function d2dm_dmf_magma(T)
    return dmf_magma(T)
end



"""
    d2dm_update_T_NG!(T, T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny, dmf_rock_arr)

Update temperature field `T` using the Newell-Georgiadis method. This function utilizes CUDA for parallel computation.

# Arguments
- `T`: Temperature field to be updated.
- `T_old`: Previous state of the temperature field.
- `T_top`: Boundary condition at the top boundary.
- `T_bot`: Boundary condition at the bottom boundary.
- `C`: Thermal conductivity of the material.
- `lam_r_rhoCp`: Coefficient related to rock properties and specific heat capacity.
- `lam_m_rhoCp`: Coefficient related to material properties and specific heat capacity.
- `L_Cp`: Length scale for thermal diffusion.
- `dx`, `dy`: Grid spacing in x and y directions.
- `dt`: Time step.
- `nx`, `ny`: Dimensions of the grid.
- `dmf_rock_arr`: Array containing rock-specific information.

# Returns
- Nothing: The function updates the temperature field `T` in place.
"""
function d2dm_update_T_NG!(T, T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny, dmf_rock_arr)
    #println("Debug - entered d2dm_update_T_NG")
    # dmf_rock_arr = CuArray{Float64,1}(undef, nx * ny)
    # dmf_rock_arr = d2dm_dmf_rock(dmf_rock_arr)

    blockSize = (28, 32)
    gridSize = (Int64(floor((nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((ny + blockSize[2] - 1) / blockSize[2])))

    @cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] d2dm_update_T!(T, T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny, dmf_rock_arr)
    synchronize()
end

"""
`d2dm_update_T!(T, T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny, dmf_rock_arr)`

Update the temperature field `T` at each grid point using finite differences.
The function takes into account the thermal conductivity and material properties to compute the heat fluxes in the x and y directions.
Boundary conditions for the top and bottom boundaries are applied.

# Arguments
- `T`: Array of temperatures.
- `T_old`: Not used but included for compatibility with other functions.
- `T_top`: Temperature at the top boundary.
- `T_bot`: Temperature at the bottom boundary.
- `C`: Array of material fractions.
- `lam_r_rhoCp`: Thermal conductivity and density times specific heat capacity for rock.
- `lam_m_rhoCp`: Thermal conductivity and density times specific heat capacity for magma.
- `L_Cp`: representing the effect of material properties on thermal diffusivity.
- `dx`, `dy`: Grid spacing in x and y directions.
- `dt`: Time step.
- `nx`, `ny`: Number of grid points in the x and y directions.
- `dmf_rock_arr`: Array of melt fractions for rocks.

# Notes
The function assumes that `idc` is a predefined function to index into 1D arrays using `(ix, iy, nx)`.
"""
function d2dm_update_T!(T, T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny, dmf_rock_arr)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    if (ix > nx - 1) || (iy > (ny - 1))
        return
    end

    qxw::Float64, qxe::Float64, qys::Float64, qyn::Float64 = 0.0, 0.0, 0.0, 0.0

    if ix == 0
        qxw = 0.0
    else
        qxw = -(T[idc(ix, iy, nx)] - T[idc(ix - 1, iy, nx)]) / dx
    end

    if ix == nx - 1
        qxe = 0.0
    else
        qxe = -(T[idc(ix + 1, iy, nx)] - T[idc(ix, iy, nx)]) / dx
    end


    if iy == 0
        qys = -2.0 * (T[idc(ix, iy, nx)] - T_bot) / dy
    else
        qys = -(T[idc(ix, iy, nx)] - T[idc(ix, iy - 1, nx)]) / dy
    end


    if iy == (ny - 1)
        qyn = -2.0 * (T_top - T[idc(ix, iy, nx)]) / dy
    else
        qyn = -(T[idc(ix, iy + 1, nx)] - T[idc(ix, iy, nx)]) / dy
    end

    #FIXME: bad interpolation
    dmf::Float64 = dmf_magma(T[idc(ix, iy, nx)]) * C[idc(ix, iy, nx)] + dmf_rock_arr[idc(ix, iy, nx)] * (1.0 - C[idc(ix, iy, nx)])
    lam_rhoCp::Float64 = (lam_m_rhoCp * C[idc(ix, iy, nx)]) + lam_r_rhoCp * (1.0 - C[idc(ix, iy, nx)])

    chi::Float64 = lam_rhoCp / (1.0 + L_Cp * dmf)

    T[idc(ix, iy, nx)] += -dt * chi * ((qxe - qxw) / dx + (qyn - qys) / dy)
    return
end


"""
    init_particles_Ph(pPh::AbstractVector{T}, ph::T, npartcl::Int) where T

Initializes particles in an array `pPh` with a value `ph` for the first `npartcl` elements.

# Arguments
- `pPh`: An abstract vector of type `T`.
- `ph`: The value to be initialized into each element of `pPh`.
- `npartcl`: Number of particles or elements in `pPh` to initialize.

# Example
```julia
init_particles_Ph(fill(0.0, 10), 5.0, 5)
```
This will set the first five elements of an array of length 10 to 5.0.
"""
function init_particles_Ph(pPh, ph, npartcl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if ip > npartcl
        return
    end

    pPh[ip] = ph
    return
end

"""
`sign(val)`: Returns the sign of a `val`.

# Arguments
- `val`: The number to determine the sign of.

# Returns
- 1 if `val` is greater than zero, -1 if `val` is less than zero, and 0 if `val` is exactly zero.
"""
function sign(val)
    return (val > zero(val)) - (val < zero(val))
end

"""
Converts Cartesian coordinates to elliptic coordinates.

This function takes a focal distance `f`, and Cartesian coordinates `(x, y)`,
and returns the corresponding elliptic coordinates `(xi, eta)`.

# Arguments
- `f`: The focal distance of the ellipse.
- `x`: The x-coordinate in Cartesian space.
- `y`: The y-coordinate in Cartesian space.

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the elliptic coordinates `(xi, eta)`.

# Examples
```julia
xi, eta = cart2ellipt(1.0, 0.5, 0.5)
println(xi, ", ", eta)  # Output will depend on the input values
```
"""
function cart2ellipt(f, x, y)
    xi_eta_1 = acosh(max(0.5 / f * (sqrt((x + f) * (x + f) + y * y) + sqrt((x - f) * (x - f) + y * y)), 1.0))
    xi_eta_2 = acos(min(max(x / (f * cosh(xi_eta_1)), -1.0), 1.0)) * sign(y)
    return xi_eta_1, xi_eta_2
end

"""
    rot2d(x, y, sb, cb)

Rotates a point `(x, y)` by an angle using the sine (`sb`) and cosine (`cb`) of the angle.
"""
function rot2d(x, y, sb, cb)
    return x * cb - y * sb, x * sb + y * cb
end

"""
Calculate parameters related to a crack in a material.

# Arguments
- `a::Float64`: Length of the crack.
- `b::Float64`: Breadth of the crack.
- `nu::Float64`: Poisson's ratio of the material.
- `G::Float64`: Shear modulus of the material.

# Returns
- A tuple containing two values:
  - The first value is a scalar representing some calculated parameter based on the input parameters.
  - The second value is another scalar calculated from the input parameters.
"""
function crack_params(a, b, nu, G)

    f::Float64 = 2 * nu * (a + b) - 2 * a - b

    return (-2 * b * G / f, 0.5 * f / (nu - 1))
end

"""
disp_inf_stress(s, st, ct, c, nu, G, shxi, chxi, seta, ceta)

Calculate the displacement components u_v at infinity due to stress state (s, st, ct) in a material with properties (c, nu, G).

# Arguments:
- `s`: Normal stress.
- `st`: Shear stress.
- `ct`: Tangential stress.
- `c`: Poisson's ratio.
- `nu`: Young's modulus.
- `G`: Shear modulus.
- `shxi`, `chxi`, `seta`, `ceta`: Coordinates and angles for the transformation.

# Returns:
- Tuple of displacement components u_v.
"""
function disp_inf_stress(s, st, ct, c, nu, G, shxi, chxi, seta, ceta)
    e2xi0 = 1.0
    s2b = 2.0 * st * ct
    c2b = 2.0 * ct * ct - 1.0
    sh2xi0 = 0.0
    ch2xi0 = 1.0
    sh2xi = 2.0 * shxi * chxi
    ch2xi = 2.0 * chxi * chxi - 1.0
    s2eta = 2.0 * seta * ceta
    c2eta = 2.0 * ceta * ceta - 1.0
    K = 3.0 - 4.0 * nu
    n = ch2xi - c2eta
    hlda = e2xi0 * c2b * (K * sh2xi - K * ch2xi + K * c2eta + ch2xi - sh2xi + c2eta) +
           K * (ch2xi - c2eta) - ch2xi - c2eta + 2.0 * ch2xi0 - 2.0 * c2b +
           2.0 * e2xi0 * (c2eta * c2b + s2eta * s2b) * (ch2xi0 * sh2xi - sh2xi0 * ch2xi)
    hldb = e2xi0 * (c2b * (K * s2eta - s2eta) + s2b * (K * ch2xi - K * c2eta + ch2xi + c2eta) -
                    2.0 * (c2eta * s2b - s2eta * c2b) * (ch2xi0 * ch2xi - sh2xi0 * sh2xi))
    u_v = (s * c / (8.0 * n * G) * (hlda * shxi * ceta + hldb * chxi * seta),
        s * c / (8.0 * n * G) * (hlda * chxi * seta - hldb * shxi * ceta))
    return u_v[1], u_v[2]
end

"""
displacements(st, ct, p, s1, s3, f, x, y, nu, G)

Calculate the displacements at a point `(x, y)` in a material under stress.

Arguments:
- `st`, `ct`: The sine and cosine of the angle between the normal to the surface and the direction of loading.
- `p`: Pressure applied on the surface.
- `s1`, `s3`: Principal stresses.
- `f`: Radius of the material.
- `x, y`: Coordinates of the point where displacement is calculated.
- `nu`: Poisson's ratio of the material.
- `G`: Shear modulus of the material.

Returns:
- The displacements `(u_v_1, u_v_2)` at the given point, rotated back to the original coordinate system.
"""
function displacements(st, ct, p, s1, s3, f, x, y, nu, G)
    x_y_1, x_y_2 = rot2d(x, y, -st, ct)
    if abs(x_y_1) < 1e-10
        x_y_1 = 1e-10
    end
    if abs(x_y_2) < 1e-10
        x_y_2 = 1e-10
    end


    xi_eta_1, xi_eta_2 = cart2ellipt(f, x_y_1, x_y_2)
    seta = sin(xi_eta_2)
    ceta = cos(xi_eta_2)
    shxi = sinh(xi_eta_1)
    chxi = cosh(xi_eta_1)
    u_v1_1, u_v1_2 = disp_inf_stress(s1 - p, st, ct, f, nu, G, shxi, chxi, seta, ceta)
    u_v2_1, u_v2_2 = disp_inf_stress(s3 - p, ct, -st, f, nu, G, shxi, chxi, seta, ceta)
    I = shxi * seta
    J = chxi * ceta
    u3 = 0.25 * p * f / G * (J * (3.0 - 4.0 * nu) - J)
    v3 = 0.25 * p * f / G * (I * (3.0 - 4.0 * nu) - I)

    u_v_1, u_v_2 = rot2d(u_v1_1 + u_v2_1 + u3, u_v1_2 + u_v2_2 + v3, st, ct)

    return -u_v_1, -u_v_2
end

"""
advect_particles_intrusion(px, py, a, b, x, y, theta, nu, G, ndykes, npartcl)

Advects particles in an intrusion model based on given parameters and displacements.

# Arguments
- `px::Array{Float64}`: Array of particle x-coordinates.
- `py::Array{Float64}`: Array of particle y-coordinates.
- `a::Float64`: Parameter related to the geometry of the dyke.
- `b::Float64`: Parameter related to the geometry of the dyke.
- `x::Float64`: X-coordinate of the intrusion point.
- `y::Float64`: Y-coordinate of the intrusion point.
- `theta::Float64`: Angle of the dyke.
- `nu::Float64`: Viscosity parameter.
- `G::Float64`: Gravitational acceleration.
- `ndykes::Int64`: Number of dykes.
- `npartcl::Int64`: Total number of particles.

# Returns
- `nothing`: The function updates the particle positions in place and returns nothing.
"""
function advect_particles_intrusion(px, py, a, b, x, y, theta, nu, G, ndykes, npartcl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if ip > npartcl
        return
    end

    p_a0_1, p_a0_2 = crack_params(a, b, nu, G)
    st = sin(theta)
    ct = cos(theta)
    u_v_1, u_v_2 = displacements(st, ct, p_a0_1, 0, 0, p_a0_2, px[ip] - x, py[ip] - y, nu, G)
    px[ip] += u_v_1
    py[ip] += u_v_2

    return nothing
end

"""
Applies a weight correction to elements in two arrays, `T` and `C`, based on weights stored in an array `wts`. This function operates in parallel using CUDA threads, where each thread processes a unique element in the arrays.

# Arguments
- `T::CuArray{Float32}`: The target array to be weight-corrected.
- `C::CuArray{Float32}`: A second target array to be weight-corrected.
- `wts::CuArray{Float32}`: An array of weights, where each element is used to divide the corresponding elements in `T` and `C`.
- `nx::Int`: The number of elements along the x-axis in the arrays.
- `ny::Int`: The number of elements along the y-axis in the arrays.

# Returns
- Nothing. The function modifies the input arrays `T` and `C` in place.

# Notes
- This function assumes that the input arrays are CUDA arrays (CuArray).
- Each thread is responsible for a unique element, determined by its block and thread indices.
- Elements outside the bounds of the array dimensions (`nx`, `ny`) or with zero weights are skipped to avoid division by zero errors.
"""
function p2g_weight!(T, C, wts, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    if (ix > nx) || (iy > ny - 1)
        return
    end

    if wts[iy*nx+ix] == 0.0
        return
    end

    T[iy*nx+ix] = T[iy*nx+ix] / wts[iy*nx+ix]
    C[iy*nx+ix] = C[iy*nx+ix] / wts[iy*nx+ix]

    return nothing
end

"""
p2g_project! - This function projects particles onto a grid using bilinear interpolation. It updates three arrays: T, C, and wts.

# Arguments
- `T::CuArray{Float64}`: Temperature array to be updated.
- `C::CuArray{Float64}`: Concentration array to be updated (or set based on pPh).
- `wts::CuArray{Float64}`: Weight array to be updated.
- `px::CuArray{Float64}`: x-coordinates of particles.
- `py::CuArray{Float64}`: y-coordinates of particles.
- `pT::CuArray{Float64}`: Temperature values at particle positions.
- `pPh::Union{Nothing, CuArray{Int32}}`: Phase array at particle positions (optional).
- `dx::Float64`: Grid spacing in x-direction.
- `dy::Float64`: Grid spacing in y-direction.
- `nx::Int64`: Number of grid points in x-direction.
- `ny::Int64`: Number of grid points in y-direction.
- `npartcl::Int64`: Total number of particles.
- `npartcl0::Int64`: Number of particles in the initial phase.

# Returns
- `nothing`

# Description
This function calculates bilinear interpolation weights for each particle based on its position and applies these weights to update arrays T, C, and wts at grid points. If pPh is provided, it updates C based on the phase information; otherwise, it sets C to 1.0 if the particle index is greater than npartcl0, or 0.0 otherwise.
"""
function p2g_project!(T, C, wts, px, py, pT, pPh, dx, dy, nx, ny, npartcl, npartcl0)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if ip > npartcl
        return
    end

    pxi = px[ip] / dx
    pyi = py[ip] / dy

    if pxi < -1 || pxi > nx || pyi < -1 || pyi > ny
        return
    end

    ix1 = min(max(Int64(floor(pxi)), 0), nx - 2)
    iy1 = min(max(Int64(floor(pyi)), 0), ny - 2)
    ix2 = ix1 + 1
    iy2 = iy1 + 1

    k11 = max(1 - abs(pxi - ix1), 0.0) * max(1 - abs(pyi - iy1), 0.0)
    k12 = max(1 - abs(pxi - ix1), 0.0) * max(1 - abs(pyi - iy2), 0.0)
    k21 = max(1 - abs(pxi - ix2), 0.0) * max(1 - abs(pyi - iy1), 0.0)
    k22 = max(1 - abs(pxi - ix2), 0.0) * max(1 - abs(pyi - iy2), 0.0)


    CUDA.atomic_add!(pointer(T, idc(ix1, iy1, nx)), k11 * pT[ip])
    CUDA.atomic_add!(pointer(T, idc(ix1, iy2, nx)), k12 * pT[ip])
    CUDA.atomic_add!(pointer(T, idc(ix2, iy1, nx)), k21 * pT[ip])
    CUDA.atomic_add!(pointer(T, idc(ix2, iy2, nx)), k22 * pT[ip])



    pC = (pPh == nothing) ? ((ip - 1) > npartcl0 ? 1.0 : 0.0) : Float64(pPh[ip])


    CUDA.atomic_add!(pointer(C, idc(ix1, iy1, nx)), k11 * pC)
    CUDA.atomic_add!(pointer(C, idc(ix1, iy2, nx)), k12 * pC)
    CUDA.atomic_add!(pointer(C, idc(ix2, iy1, nx)), k21 * pC)
    CUDA.atomic_add!(pointer(C, idc(ix2, iy2, nx)), k22 * pC)


    CUDA.atomic_add!(pointer(wts, idc(ix1, iy1, nx)), k11)
    CUDA.atomic_add!(pointer(wts, idc(ix1, iy2, nx)), k12)
    CUDA.atomic_add!(pointer(wts, idc(ix2, iy1, nx)), k21)
    CUDA.atomic_add!(pointer(wts, idc(ix2, iy2, nx)), k22)

    return nothing
end

"""
`d2dm_g2p!(T, T_old, px, py, pT, dx, dy, pic_amount, nx, ny, npartcl)`

Updates the temperature of particles in a 2D model based on their position and neighboring grid temperatures. The function interpolates the grid temperature at the particle's position using bilinear interpolation and then updates the particle's temperature based on this interpolated value.

# Arguments
- `T`: Array containing the current grid temperatures.
- `T_old`: Array containing the previous grid temperatures.
- `px`, `py`: Arrays containing the x and y coordinates of each particle, respectively.
- `pT`: Array containing the current temperatures of each particle. This is updated by the function.
- `dx`, `dy`: Spacings between grid points in the x and y directions, respectively.
- `pic_amount`: A factor that determines how much to weight the temperature contribution from the interpolated grid value versus the previous particle temperature.
- `nx`, `ny`: Dimensions of the grid in the x and y directions, respectively.
- `npartcl`: Total number of particles.

# Returns
- `nothing`: The function modifies arrays in place and does not return any values.
"""
function d2dm_g2p!(T, T_old, px, py, pT, dx, dy, pic_amount, nx, ny, npartcl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if ip > (npartcl)
        return nothing
    end

    #xi and xy coordinate of particle
    pxi = px[ip] / dx
    pyi = py[ip] / dy

    #check if we out of bounds and getting boundaries of cell around particle
    ix1 = min(max(Int64(floor(pxi)), 0), nx - 2)
    iy1 = min(max(Int64(floor(pyi)), 0), ny - 2)
    ix2 = ix1 + 1
    iy2 = iy1 + 1

    x1 = Float64(ix1) * dx
    x2 = Float64(ix2) * dx
    y1 = Float64(iy1) * dy
    y2 = Float64(iy2) * dy

    #T_pic - current temperature of particle?
    T_pic = blerp(x1, x2, y1, y2, T[idc(ix1, iy1, nx)], T[idc(ix1, iy2, nx)], T[idc(ix2, iy1, nx)], T[idc(ix2, iy2, nx)], px[ip], py[ip])
    T_flip = pT[ip] + T_pic - blerp(x1, x2, y1, y2, T_old[idc(ix1, iy1, nx)], T_old[idc(ix1, iy2, nx)], T_old[idc(ix2, iy1, nx)], T_old[idc(ix2, iy2, nx)], px[ip], py[ip])
    #if pic_amount == 1, pT defined by T of grid, if pic_amount == 0, T defined by dT of interpolated pT
    pT[ip] = T_pic * pic_amount + T_flip * (1.0 - pic_amount)

    return nothing
end

"""
assignUniqueLables(mf, L, tsh, nx, ny)

Assigns unique labels to each element in the matrix `mf` based on a threshold `tsh`. The function operates within a 2D grid defined by `nx` and `ny`, and uses shared memory for computation. If an element's value in `mf` is greater than or equal to `tsh`, it assigns a unique label; otherwise, it assigns `-1`.

# Arguments:
- `mf::Vector{Float64}`: Input matrix where each element represents a data point.
- `L::Vector{Int32}`: Output vector where labels are stored.
- `tsh::Float64`: Threshold value used to determine if an element should be labeled.
- `nx::Int32`: Number of elements along the x-axis.
- `ny::Int32`: Number of elements along the y-axis.

# Returns:
- Nothing. The function modifies the output vector `L` in place.
"""
function assignUniqueLables(mf, L, tsh, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    if (ix > nx - 1 || iy > ny - 1)
        return
    end

    if mf[(iy)*nx+ix+1] >= tsh
        L[(iy)*nx+ix+1] = (iy) * nx + ix
    else
        L[(iy)*nx+ix+1] = -1
    end
    return nothing
end

"""
    cwLabel!(L::Array{Int, 1}, nx::Int, ny::Int)

This function processes an array `L` representing a label map in a 2D grid of size `nx` x `ny`.
It iterates over the array in reverse order along the x-axis for each row.
If two consecutive elements are both greater than or equal to 1, it sets the left element to match the right element.

# Arguments
- `L::Array{Int, 1}`: A one-dimensional array representing a label map.
- `nx::Int`: The number of columns in the grid.
- `ny::Int`: The number of rows in the grid.

# Returns
- Nothing; modifies `L` in place.
"""
function cwLabel!(L, nx, ny)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if iy > ny
        return
    end

    for ix = nx-1:-1:1
        if L[(iy-1)*nx+ix] >= 1 && L[(iy-1)*nx+ix+1] >= 1
            L[(iy-1)*nx+ix] = L[(iy-1)*nx+ix+1]
        end
    end
end

"""
find_root(L, idx)

Given an array `L` and an index `idx`, this function finds the root of the element at index `idx`.
The root is defined as the index where `L[label] == label - 1`.

# Arguments:
- L::Array{Int32}: An array of integers.
- idx::Int32: The index to start from.

# Returns:
- Int32: The root index found.

# Examples:
```julia
julia> find_root([1, 0, 2, 3], 2)
1
```
"""
function find_root(L, idx)
    label::Int32 = idx
    while L[label] != (label - 1)
        label = L[label] + 1
    end
    return label - 1
end

"""
merge_labels!(L::Array{Int32}, div::Int64, nx::Int64, ny::Int64)

Merges labels in a 2D array `L` along rows based on a division factor `div`. This function is typically used in image processing or grid-based simulations to merge neighboring regions labeled with non-negative integers.

- `L`: A 1D array representing the 2D grid where each element contains a label.
- `div`: The size of the division along the row axis.
- `nx`: The number of columns in the grid.
- `ny`: The number of rows in the grid.

The function iterates over each block and thread, merging labels between neighboring rows if both labels are non-negative. It uses a helper function `find_root` to find the root label for a given index.
"""
function merge_labels!(L, div, nx, ny)
    iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = Int64(floor(div / 2)) + iy * div - 1
    if iy > (ny - 2)
        return
    end

    for ix = 0:(nx-1)
        if (L[iy*nx+ix+1] >= 0) && (L[(iy+1)*nx+ix+1] >= 0)
            lroot::Int32 = find_root(L, iy * nx + ix + 1)
            rroot::Int32 = find_root(L, (iy + 1) * nx + ix + 1)
            L[min(lroot, rroot)+1] = L[max(lroot, rroot)+1]
        end
    end

    return nothing
end

"""
# relabel(L, nx, ny)

Relabels elements in a lattice `L` of size `nx x ny`. This function processes elements in parallel using CUDA's grid and block threading model.

## Arguments:
- `L::Array{Int64}`: The input lattice as an array of integers.
- `nx::Int64`: The number of columns in the lattice.
- `ny::Int64`: The number of rows in the lattice.

## Returns:
- `nothing`: The function modifies the input lattice `L` in place and does not return any value.

## Notes:
- The function uses CUDA's grid and block threading model to parallelize the processing of elements in the lattice.
- Elements with a non-negative value are processed by calling `find_root(L, iy * nx + ix + 1)` to potentially update their values in the lattice.
- Only elements within valid indices `[0, nx*ny-1]` are considered.
"""
function relabel(L, nx, ny)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    if (ix > nx - 1) || (iy > ny - 1)
        return
    end

    if L[iy*nx+ix+1] >= 0
        L[iy*nx+ix+1] = find_root(L, iy * nx + ix + 1)
    end

    return nothing
end

"""
Advect particles based on an eruption model.

This function updates the positions of particles in a 2D grid according to an eruption model. It calculates the velocity components `u` and `v` for each particle, which are then used to update the particle's position.

# Arguments
- `px::Array{Float64}`: Array containing the x-coordinates of the particles.
- `py::Array{Float64}`: Array containing the y-coordinates of the particles.
- `idx::Array{Int32}`: Index array representing the grid cells.
- `gamma::Float64`: Eruption parameter that affects the velocity calculation.
- `dxl::Float64`: Grid spacing in the x-direction.
- `dyl::Float64`: Grid spacing in the y-direction.
- `npartcl::Int32`: Number of particles.
- `ncells::Int32`: Number of grid cells.
- `nxl::Int32`: Number of grid cells in the x-direction.
- `nyl::Int32`: Number of grid cells in the y-direction.

# Returns
- `nothing`: The function updates the positions of particles in place and does not return any value.
"""
function advect_particles_eruption(px, py, idx, gamma, dxl, dyl, npartcl, ncells, nxl, nyl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1

    if ip > npartcl - 1
        return
    end

    u = 0.0
    v = 0.0

    for i = 0:ncells-1
        ic = idx[i+1]
        icx = ic % nxl
        icy = ic ÷ nyl

        xl = icx * dxl
        yl = icy * dyl

        dxl2 = dxl * dxl
        dyl2 = dyl * dyl

        delx = px[ip+1] - xl
        dely = py[ip+1] - yl
        r = max(sqrt(delx * delx + dely * dely), sqrt(dxl2 + dyl2))
        r2_2pi = r * r * 2 * π

        u -= dxl2 * (1.0 - gamma) * delx / r2_2pi
        v -= dyl2 * (1.0 - gamma) * dely / r2_2pi
    end

    px[ip+1] += u
    py[ip+1] += v
    return nothing
end

"""
	average!(mfl, T, C, nl, nx, ny)

averaging melt fraction based on T and ration of magma to the host rock

# Arguments
- `mfl`: melt fraction grid, [1]
- `T`: Temperature grid, [°C]
- `C`: Grid with the ratio of magma to the (host rock + magma) in cell, [1]
- `nl`: grid reduction factor, [1]
- `nx`: x-axis grid resolution, [1]
- `ny`: y-axis grid resolution, [1]
"""
function average!(mfl, T, C, nl, nx, ny, mf_rock_c)
    ixl = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iyl = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    if ixl > (nx ÷ nl - 1) || iyl > (ny ÷ nl - 1)
        return
    end

    avg = 0.0
    for ix = (ixl*nl):((ixl+1)*nl-1)
        if ix > nx - 1
            break
        end
        for iy = (iyl*nl):((iyl+1)*nl-1)
            if iy > ny - 1
                break
            end
            vf = C[iy*nx+ix+1]
            avg = avg + (mf_magma(T[(iy*nx+ix+1)])) * vf + mf_rock_c[iy*nx+ix+1] * (1 - vf)
        end
    end
    avg /= (nl * nl)
    mfl[iyl*(nx÷nl)+ixl+1] = avg
    return nothing
end

"""
count_particles! increments the count of particles in a 2D grid using global coordinates.

# Arguments
- `pcnt`: A pointer to the array where particle counts are stored.
- `px`, `py`: Arrays containing the x and y coordinates of each particle, respectively.
- `dx`, `dy`: The spatial resolution or bin size in the x and y directions.
- `nx`, `ny`: The number of bins in the x and y directions.
- `npartcl`: The total number of particles.

# Returns
- Nothing

This function is designed to be used with CUDA.jl for parallel processing on GPU.
"""
function count_particles!(pcnt, px, py, dx, dy, nx, ny, npartcl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if ip > npartcl
        return
    end

    pxi = px[ip] / dx
    pyi = py[ip] / dy

    ix = min(max(Int64(floor(pxi)), 0), nx - 2)
    iy = min(max(Int64(floor(pyi)), 0), ny - 2)

    CUDA.atomic_add!(pointer(pcnt, (iy * nx + ix + 1)), Int32(1))

    return nothing
end

"""
inject_particles!(px::CuArray{Float32}, py::CuArray{Float32}, pT::CuArray{Float32},
                pPh::CuArray{Int32}, npartcl::CuDeviceArray{Int32, 1}, pcnt::CuDeviceArray{Int32, 1},
                T::CuDeviceArray{Float32, 2}, C::CuDeviceArray{Float32, 2}, dx::Float32, dy::Float32,
                nx::Int32, ny::Int32, min_pcount::Int32, max_npartcl::Int32)

Injects particles into the simulation grid based on certain conditions.

# Arguments
- `px`: Particle x-coordinates.
- `py`: Particle y-coordinates.
- `pT`: Particle temperatures.
- `pPh`: Particle phases.
- `npartcl`: Array to store the number of particles.
- `pcnt`: Array with particle count per grid cell.
- `T`: Temperature field.
- `C`: Concentration field.
- `dx`, `dy`: Grid spacing in x and y directions.
- `nx`, `ny`: Number of grid points in x and y directions.
- `min_pcount`: Minimum particle count per grid cell to trigger injection.
- `max_npartcl`: Maximum number of particles allowed.

# Notes
- This function is designed for GPU execution using CUDA.jl.
- It operates within a 2D block and thread hierarchy typical of CUDA grids.
"""
function inject_particles!(px, py, pT, pPh, npartcl, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

    if ix > nx - 2 || iy > ny - 2
        return
    end

    if pcnt[iy*nx+ix+1] < min_pcount
        for ioy = 0:1
            for iox = 0:1
                inx = ix + iox
                iny = iy + ioy
                new_npartcl = CUDA.atomic_add!(pointer(npartcl, 1), Int32(1))
                if new_npartcl > max_npartcl - 1
                    break
                end
                ip = new_npartcl + 1
                px[ip] = dx * inx
                py[ip] = dy * iny

                pT[ip] = T[idc(inx, iny, nx)]
                pPh[ip] = C[idc(inx, iny, nx)] < 0.5 ? 0 : 1
            end
        end
    end
    return nothing
end


"""
ccl(mf::CuArray{Float32}, L::CuArray{Int32}, tsh::Float32, nx::Int32, ny::Int32)

This function performs connected component labeling (CCL) on a 2D array `mf` using CUDA. The labels are stored in the 2D array `L`. Only points where `mf` exceeds the threshold `tsh` are considered part of components.

# Arguments
- `mf::CuArray{Float32}`: A 2D array representing the input image data.
- `L::CuArray{Int32}`: A 2D array to store the labels of connected components.
- `tsh::Float32`: The threshold value for labeling points in `mf`.
- `nx::Int32`: The number of columns in the 2D array `mf` and `L`.
- `ny::Int32`: The number of rows in the 2D array `mf` and `L`.

# Returns
- Nothing. The labels are stored directly in `L`.
"""
function ccl(mf, L, tsh, nx, ny)
    blockSize2D = (28, 32)
    gridSize2D = ((nx + blockSize2D[1] - 1) ÷ blockSize2D[1], (ny + blockSize2D[2] - 1) ÷ blockSize2D[2])

    #add lables to the points where ms > tsh
    CUDA.@sync begin
        @cuda blocks = gridSize2D threads = blockSize2D assignUniqueLables(mf, L, tsh, nx, ny)
    end
    blockSize1D = dev_thread
    gridSize1D = (ny + blockSize1D - 1) ÷ blockSize1D

    #merge labels in components horisotally
    CUDA.@sync begin
        @cuda blocks = gridSize1D threads = blockSize1D cwLabel!(L, nx, ny)
    end

    div = 2
    npw = Int64(ceil(log2(ny)))
    nyw = 1 << npw
    for i = 0:(npw-1)
        gridSize1D = Int64(floor(max((nyw + blockSize1D - 1) / blockSize1D / div, 1)))

        #merge roots vertically
        CUDA.@sync begin
            @cuda blocks = gridSize1D threads = blockSize1D merge_labels!(L, div, nx, ny)
        end

        div = div * 2
    end

    #relable based on roots
    CUDA.@sync begin
        @cuda blocks = gridSize2D threads = blockSize2D relabel(L, nx, ny)
    end

    return nothing
end

"""
    init_particles_T(pT, T_magma, npartcl)

Initialize particles with a given temperature `T_magma` within an array `pT`.

# Arguments
- `pT`: An array where the temperatures of the particles will be stored.
- `T_magma`: The temperature to be set for each particle.
- `npartcl`: The total number of particles.

# Notes
This function assumes that the thread block and thread ID are correctly set up in the kernel execution environment.
"""
function init_particles_T(pT, T_magma, npartcl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (ip > (npartcl))
        return
    end

    pT[ip] = T_magma
    return
end

"""
Initialize particles in `pT` array to a given temperature `T_magma`.

# Arguments
- `pT::Array{T, 1}`: Array where particle temperatures will be stored.
- `T_magma::T`: Temperature of magma to initialize particles with.
- `npartcl::Int`: Number of particles in the array.

# Returns
- Nothing. The function modifies `pT` in place.
"""
function init_particles_T_magma(pT, T_magma, npartcl)
    ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (ip > (npartcl))
        return
    end

    pT[ip] = T_magma
    return
end

"""
Write data to an HDF5 file.

# Arguments
- `filename::String`: The name of the file to write.
- `data`: The data to write to the file.

# Returns
- None
"""
function write_h5(filename, data)
    file = joinpath(dir, "$(filename)")
    open(file, "w") do fid
        write(fid, data)
    end
end

"""
    write_h5(filename, data)

Write data to an HDF5 file.

# Arguments
- `filename::String`: The name of the file to write.
- `data`: Data to be written to the file.

# Examples
```julia
write_h5("example.h5", some_data)
```
"""
function write_h5(filename, data)
    file = joinpath(dir, "$(filename)")
    open(file, "w") do fid
        write(fid, data)
    end
end

"""
small_mailbox_out(filename::String, T::Array{Float64}, pT::Array{Float64}, C::Array{Float64}, mT::Array{Float64}, staging::Array{Int32}, L::Array{Float64}, nx::Int32, ny::Int32, nxl::Int32, nyl::Int32, max_npartcl::Int32, max_nmarker::Int32, px::Array{Float64}, py::Array{Float64}, mx::Array{Float64}, my::Array{Float64}, h_px_dykes::Array{Float64}, pcnt::Array{Int32}, mfl::Array{Int32}, dx::Float64, dy::Float64, Lx::Float64, Ly::Float64)

Write the results of a simulation to an HDF5 file. The function takes in arrays and parameters representing various state variables and geometry of the simulation domain.

# Arguments
- `filename`: Name of the file to write the data to.
- `T`: Array containing temperature values.
- `pT`: Array containing pressure values.
- `C`: Array containing concentration values.
- `mT`: Array containing mass values.
- `staging`: Array indicating the staging of particles or markers.
- `L`: Array containing length values for the domain.
- `nx`, `ny`: Dimensions of the grid.
- `nxl`, `nyl`: Lengths along x and y directions in the grid.
- `max_npartcl`, `max_nmarker`: Maximum number of particles and markers respectively.
- `px`, `py`: Arrays containing the positions of particles and markers.
- `mx`, `my`: Arrays containing additional marker properties.
- `h_px_dykes`: Array containing additional properties related to dykes.
- `pcnt`, `mfl`: Arrays with particle and marker flags.
- `dx`, `dy`: Grid spacing in x and y directions.
- `Lx`, `Ly`: Total length and width of the simulation domain.

# Returns
- None. The function writes data directly to a file.
"""
function small_mailbox_out(filename, T, pT, C, mT, staging, L, nx, ny, nxl, nyl, max_npartcl, max_nmarker, px, py, mx, my, h_px_dykes, pcnt, mfl, dx, dy, Lx, Ly)
    @time begin
        #@printf("%s writing results to disk  | ", bar2)
        #filename = "grid." * string(it) * ".h5"

        if isfile(filename)
            rm(filename)
        end

        fid = h5open(filename, "w")
        #h5write(filename, "T", T)
        #h5write(filename, "C", C)

        h_T = Array{Float64,1}(undef, nx * ny)#array of double values from matlab script
        h_C = Array{Float64,1}(undef, nx * ny)#array of double values from matlab script

        copyto!(h_T, T)
        copyto!(h_C, C)

        write(fid, "T", h_T)
        write(fid, "C", h_C)

        write(fid, "nx", nx)
        write(fid, "ny", ny)

        write(fid, "dx", dx)
        write(fid, "dy", dy)

        write(fid, "Lx", Lx)
        write(fid, "Ly", Ly)

        close(fid)
    end
end

"""
`d2dm_make_snapshot(vp, gp, filename, FLAG_make_snapshot)`

Save the state of variables `vp` and `gp` to an HDF5 file if `FLAG_make_snapshot` is true.

# Arguments
- `vp`: An object containing velocity model parameters.
- `gp`: An object containing grid parameters, which may include CuArray fields that need special handling.
- `filename`: The name of the HDF5 file where the snapshot will be saved.
- `FLAG_make_snapshot`: A boolean flag indicating whether to create a snapshot.

# Behavior
If `FLAG_make_snapshot` is true, this function will open an HDF5 file at the specified `filename`, write out all fields from both `vp` and `gp` objects, handling CuArray types appropriately by copying them to CPU memory. Each field's name is used as the dataset name in the HDF5 file.

# Examples
```julia
vp = VelocityModel(...)
gp = GridParameters(...)
d2dm_make_snapshot(vp, gp, "snapshot.hdf5", true)
```
"""
function d2dm_make_snapshot(vp, gp, filename, FLAG_make_snapshot)
    if (FLAG_make_snapshot)
        #filename_donwload = @sprintf("d2d_snapshot_%d_%s.hdf5",vp.it, Dates.format(now(), "yyyy_mm_dd_HH_MM_SS"))
        #filename_donwload = @sprintf("d2d_snapshot.hdf5")
        fid = h5open(filename, "w")

        for n in fieldnames(typeof(vp))
            println(getfield(vp, n))
            write(fid, string(n), getfield(vp, n))
        end

        for n in fieldnames(typeof(gp))
            if (getfield(gp, n) isa CuArray)
                d2d_cu_type = eltype(getfield(gp, n))

                nn::Array{d2d_cu_type,1} = Array{d2d_cu_type,1}(undef, size(getfield(gp, n))[1])
                copyto!(nn, getfield(gp, n))

                #println(getfield(gp,n))
                write(fid, string(n), nn)
                println("sucess!!")
            else
                #println(getfield(gp,n))
                write(fid, string(n), getfield(gp, n))
            end
        end

        println("snapshot saved to " * filename)
        #log_to_buffer("snapshot saved to " * filename)

        close(fid)
    end
end

"""
    mailbox_out(filename, T, pT, C, mT, staging, L, nx, ny, nxl, nyl, max_npartcl, max_nmarker, px, py, mx, my, h_px_dykes, pcnt, mfl)

Write simulation data to an HDF5 file.

# Arguments
- `filename::String`: The name of the file where data will be saved.
- `T::Array{Float64,1}`: Array containing temperature values.
- `pT::Array{Float64,1}`: Array containing particle temperature values.
- `C::Array{Float64,1}`: Array containing concentration values.
- `mT::Array{Float64,1}`: Array containing marker temperature values.
- `staging::String`: Staging information.
- `L::Array{Int32,1}`: Array containing lattice values.
- `nx::Int32`: Number of grid points in x-direction.
- `ny::Int32`: Number of grid points in y-direction.
- `nxl::Int32`: Number of lattice points in x-direction.
- `nyl::Int32`: Number of lattice points in y-direction.
- `max_npartcl::Int32`: Maximum number of particles.
- `max_nmarker::Int32`: Maximum number of markers.
- `px::Array{Float64,1}`: Array containing x-coordinates of particles.
- `py::Array{Float64,1}`: Array containing y-coordinates of particles.
- `mx::Array{Float64,1}`: Array containing x-coordinates of markers.
- `my::Array{Float64,1}`: Array containing y-coordinates of markers.
- `h_px_dykes::Float64`: Height of the Dykes in pixels.
- `pcnt::Array{Int32,1}`: Array containing particle count values.
- `mfl::Array{Float64,1}`: Array containing marker flow values.

# Description
This function writes simulation data to an HDF5 file. It first checks if the file exists and removes it if it does. Then, it creates a new HDF5 file and writes various arrays containing simulation data into it.
"""
function mailbox_out(filename, T, pT, C, mT, staging, L, nx, ny, nxl, nyl, max_npartcl, max_nmarker, px, py, mx, my, h_px_dykes, pcnt, mfl)
    @time begin
        bar1 = "├──"
        bar2 = "\t ├──"
        #@printf("%s writing results to disk  | ", bar2)
        #filename = "grid." * string(it) * ".h5"

        if isfile(filename)
            rm(filename)
        end

        fid = h5open(filename, "w")
        #h5write(filename, "T", T)
        #h5write(filename, "C", C)

        h_pcnt = Array{Int32,1}(undef, nx * ny)#array of double values from matlab script
        h_T = Array{Float64,1}(undef, nx * ny)#array of double values from matlab script
        h_C = Array{Float64,1}(undef, nx * ny)#array of double values from matlab script
        h_pT = Array{Float64,1}(undef, max_npartcl)#array of double values from matlab script
        h_mT = Array{Float64,1}(undef, max_nmarker)#array of double values from matlab script
        h_L = Array{Int32,1}(undef, nxl * nyl)#array of double values from matlab script
        h_px = Array{Float64,1}(undef, max_npartcl)#array of double values from matlab script
        h_py = Array{Float64,1}(undef, max_npartcl)#array of double values from matlab script
        h_mx = Array{Float64,1}(undef, max_nmarker)#array of double values from matlab script
        h_my = Array{Float64,1}(undef, max_nmarker)#array of double values from matlab script
        h_mfl = Array{Float64,1}(undef, nxl * nyl)#array of double values from matlab script

        copyto!(h_pcnt, pcnt)
        copyto!(h_T, T)
        copyto!(h_pT, pT)
        copyto!(h_mT, mT)
        copyto!(h_C, C)

        copyto!(h_px, px)
        copyto!(h_py, py)
        copyto!(h_mx, mx)
        copyto!(h_my, my)
        copyto!(h_mfl, mfl)

        write(fid, "pcnt", h_pcnt)
        write(fid, "T", h_T)
        write(fid, "pT", h_pT)
        write(fid, "mT", h_mT)
        write(fid, "C", h_C)

        write(fid, "px", h_px)
        write(fid, "py", h_py)
        write(fid, "mx", h_mx)
        write(fid, "my", h_my)
        write(fid, "px_dykes", h_px_dykes)
        write(fid, "mfl", h_mfl)
        #write(fid, "L", h_L)

        copyto!(h_L, L)
        write(fid, "L", h_L)

        close(fid)
    end
end


"""
Generate a random number within a specified range using a normal distribution.

# Arguments:
- `u`: The mean of the normal distribution.
- `d`: The standard deviation of the normal distribution.
- `dyke_type::String`: A string representing the type of dyke, which is not used in the function but included for documentation purposes.

# Returns:
- A random number drawn from a normal distribution with mean `u` and standard deviation `d`, constrained to be within the range (0, 1).
"""
function rand_limited(u, d, dyke_type::String)
    while ((ans <= 0) || (ans >= 1))
        ans = rand(Normal(u, d), 1)[1]
    end

    return ans
end

"""
Generate a random value within specified limits based on the given distribution type.

# Arguments
- `u`: The mean or mode of the distribution.
- `d`: The standard deviation for Normal and LogNormal distributions, scale for Uniform distribution.
- `dyke_type::String`: The type of distribution to use ("Normal", "Uniform", "LogNormal").

# Returns
- A random value generated from the specified distribution within the valid range (0, 1).
"""
function rand_limited_2(u, d, dyke_type::String)
    ans::Float64 = -1
    while ((ans <= 0) || (ans >= 1))
        if (dyke_type == "Normal")
            ans = rand(Normal(u, d), 1)[1]
        elseif (dyke_type == "Uniform")
            ans = rand(Uniform(), 1)[1]
        elseif (dyke_type == "LogNormal")
            ans = rand(LogNormal(u, d), 1)[1]
        end
    end
    return ans
end

"""
`d2dm_read_params(gp::GridParams, vp::VarParams, data_folder) -> Nothing`

Reads parameters from binary and HDF5 files to initialize simulation parameters.

# Arguments:
- `gp::GridParams`: Structure holding grid-related parameters.
- `vp::VarParams`: Structure holding variable parameters.
- `data_folder::String`: Path to the directory containing the input data files.

# Description:
This function reads various parameters required for a 2D model from binary (`pa.bin`, `dykes.bin`)
and HDF5 files. It populates structures `gp` and `vp` with these parameters, setting up the initial
conditions for the simulation.
"""
function d2dm_read_params(gp::GridParams, vp::VarParams, data_folder)

    dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
    ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script

    io = open(data_folder * "pa.bin", "r")
    read!(io, dpa)
    read!(io, ipa)

    ipar = 1
    vp.Lx, ipar = read_par(dpa, ipar)
    vp.Ly, ipar = read_par(dpa, ipar)
    vp.lam_r_rhoCp, ipar = read_par(dpa, ipar)
    vp.lam_m_rhoCp, ipar = read_par(dpa, ipar)
    vp.L_Cp, ipar = read_par(dpa, ipar)
    vp.T_top, ipar = read_par(dpa, ipar)
    vp.T_bot, ipar = read_par(dpa, ipar)
    vp.T_magma, ipar = read_par(dpa, ipar)
    vp.tsh, ipar = read_par(dpa, ipar)
    vp.gamma, ipar = read_par(dpa, ipar)
    vp.Ly_eruption, ipar = read_par(dpa, ipar)
    vp.nu, ipar = read_par(dpa, ipar)
    vp.G, ipar = read_par(dpa, ipar)
    vp.dt, ipar = read_par(dpa, ipar)
    vp.dx, ipar = read_par(dpa, ipar)
    vp.dy, ipar = read_par(dpa, ipar)
    vp.eiter, ipar = read_par(dpa, ipar)
    vp.pic_amount, ipar = read_par(dpa, ipar)
    tfin, ipar = read_par(dpa, ipar)

    ipar = 1

    vp.pmlt, ipar = read_par(ipa, ipar)
    vp.nx, ipar = read_par(ipa, ipar)
    vp.ny, ipar = read_par(ipa, ipar)
    vp.nl, ipar = read_par(ipa, ipar)
    vp.nt, ipar = read_par(ipa, ipar)
    vp.niter, ipar = read_par(ipa, ipar)
    vp.nout, ipar = read_par(ipa, ipar)
    vp.nsub, ipar = read_par(ipa, ipar)
    vp.nerupt, ipar = read_par(ipa, ipar)
    vp.npartcl, ipar = read_par(ipa, ipar)
    vp.nmarker, ipar = read_par(ipa, ipar)
    vp.nSample, ipar = read_par(ipa, ipar)

    gp.critVol = Array{Float64,1}(undef, vp.nSample) #???#Critical volume when eruption appears, predefined variable
    read!(io, gp.critVol)

    #array 0 0 1 0 0 ... like, where 1 -instrusion
    gp.ndykes = Array{Int32,1}(undef, vp.nt)#number of dykes intruded on n-th time step
    read!(io, gp.ndykes)

    ndykes_all = 0

    #count all dykes
    for istep in 1:vp.nt
        ndykes_all = ndykes_all + gp.ndykes[istep]
    end

    #array which describes amount of particles in new dyke
    gp.particle_edges = Array{Int32,1}(undef, ndykes_all + 1)
    read!(io, gp.particle_edges)

    gp.marker_edges = Array{Int32,1}(undef, ndykes_all + 1)
    read!(io, gp.marker_edges)

    close(io)

    cap_frac = 3  #value to spcify how much particles we allow to inject in runtime
    vp.npartcl0 = vp.npartcl #initial amount of particles
    vp.max_npartcl = convert(Int64, vp.npartcl * cap_frac) + gp.particle_edges[ndykes_all+1] #???#count max particles
    println(vp.npartcl)
    println(gp.particle_edges[ndykes_all+1])
    println("max_npartcl")
    println(vp.max_npartcl)
    nmarker0 = vp.nmarker

    vp.max_nmarker = vp.nmarker + gp.marker_edges[ndykes_all+1]


    #blockSize(28, 32);
    #gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);


    blockSize = (28, 32)
    gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

    gp.T = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.T_old = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.C = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.wts = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.pcnt = CuArray{Int32,1}(undef, vp.nx * vp.ny)

    #	gp.a = CuArray{Float64}(undef, (1, 2))

    gp.px = CuArray{Float64}(undef, vp.max_npartcl)#x coordinate of particle
    gp.py = CuArray{Float64}(undef, vp.max_npartcl)#y coordinate of particle
    gp.pT = CuArray{Float64}(undef, vp.max_npartcl)#Temperature of particle
    gp.pPh = CuArray{Int8}(undef, vp.max_npartcl)#???#Ph?

    np_dykes = gp.particle_edges[ndykes_all+1]#number of particles in each dyke during intrusion

    gp.px_dykes = CuArray{Float64,1}(undef, np_dykes)#x of dykes particles
    gp.py_dykes = CuArray{Float64,1}(undef, np_dykes)#y of dykes particles

    gp.mx = CuArray{Float64,1}(undef, vp.max_nmarker)#x of marker
    gp.my = CuArray{Float64,1}(undef, vp.max_nmarker)#y of marker
    gp.mT = CuArray{Float64,1}(undef, vp.max_nmarker)#T of marker

    gp.staging = Array{Float64,1}(undef, vp.max_npartcl)
    gp.npartcl_d = CuArray{Int32,1}(undef, 1)
    gp.npartcl_h = Array{Int32,1}(undef, 1)


    #small grid dimensions
    nxl = convert(Int64, vp.nx / vp.nl)
    nyl = convert(Int64, vp.ny / vp.nl)

    vp.nxl = nxl
    vp.nyl = nyl

    #small grid itself
    gp.L = CuArray{Int32,1}(undef, nxl * nyl)

    #small grid on host
    gp.L_host = Array{Int32,1}(undef, nxl * nyl)

    #???
    gp.mfl = CuArray{Float64,1}(undef, nxl * nyl)


    #a and b of ellips for dykes
    gp.dyke_a = Array{Float64,1}(undef, ndykes_all)
    gp.dyke_b = Array{Float64,1}(undef, ndykes_all)

    #x and y coordinate of center
    gp.dyke_x = Array{Float64,1}(undef, ndykes_all)
    gp.dyke_y = Array{Float64,1}(undef, ndykes_all)

    #???
    gp.dyke_t = Array{Float64,1}(undef, ndykes_all)

    #NOTE:Dykes data upload takes time
    io = open(data_folder * "dykes.bin", "r")
    read!(io, gp.dyke_a)
    read!(io, gp.dyke_b)
    read!(io, gp.dyke_x)
    read!(io, gp.dyke_y)
    read!(io, gp.dyke_t)

    close(io)

    fid = h5open(data_folder * "particles.h5", "r")

    gp.h_px = Array{Float64,1}(undef, vp.max_npartcl)
    gp.h_py = Array{Float64,1}(undef, vp.max_npartcl)

    gp.h_px = read(fid, "px")
    gp.h_py = read(fid, "py")

    copyto!(gp.px, gp.h_px)
    copyto!(gp.py, gp.h_py)

    #???
    gp.h_px_dykes = Array{Float64,1}(undef, np_dykes)
    gp.h_py_dykes = Array{Float64,1}(undef, np_dykes)

    gp.h_px_dykes = read(fid, "px_dykes")
    gp.h_py_dykes = read(fid, "py_dykes")

    copyto!(gp.px_dykes, gp.h_px_dykes)
    copyto!(gp.py_dykes, gp.h_py_dykes)

    close(fid)

    #process markers
    fid = h5open(data_folder * "markers.h5", "r")

    obj = fid["0"]

    gp.h_mx = Array{Float64,1}(undef, vp.max_nmarker)
    gp.h_my = Array{Float64,1}(undef, vp.max_nmarker)
    gp.h_mT = Array{Float64,1}(undef, vp.max_nmarker)

    gp.h_mx = read(obj, "mx")
    gp.h_my = read(obj, "my")
    gp.h_mT = read(obj, "mT")

    close(fid)

    copyto!(gp.mx, gp.h_mx)
    copyto!(gp.my, gp.h_my)
    copyto!(gp.mT, gp.h_mT)


    NDIGITS = Int32(floor(log10(vp.nt))) + 1

    filename = data_folder * "grid." * "0"^NDIGITS * ".h5"
    println(vp.nx)
    println(vp.ny)
    fid = h5open(filename, "r")
    T_h = read(fid, "T")
    copyto!(gp.T, T_h)
    C_h = read(fid, "C")
    copyto!(gp.C, C_h)
    close(fid)

    iSample = Int32(1)

end


"""
d2dm_init(gp::GridParams, vp::VarParams, markers_flag)

Initializes the simulation by setting up grid and particle parameters,
and updating marker temperatures if `markers_flag` is true.
- `gp`: Grid parameters containing temperature arrays (`T`, `T_old`) and position arrays (`px`, `py`, `pT`).
- `vp`: Variable parameters including number of particles, markers, temperature thresholds, etc.
- `markers_flag`: Boolean flag to determine if marker temperatures should be initialized.

This function updates the simulation grid based on particle positions and initializes new particles or markers
with specific temperature conditions. It also synchronizes GPU operations for parallel execution.
"""
function d2dm_init(gp::GridParams, vp::VarParams, markers_flag)
    pic_amount_tmp = vp.pic_amount
    vp.pic_amount = 1.0

    blockSize1D = dev_thread
    gridSize1D = convert(Int64, floor((vp.npartcl + blockSize1D - 1) / blockSize1D))

    #NOTE:
    #changing only pT
    #grid to particles interpolation
    #differene with cuda like 6.e-8 for some reason
    @cuda blocks = gridSize1D threads = blockSize1D d2dm_g2p!(gp.T, gp.T_old, gp.px, gp.py, gp.pT, vp.dx, vp.dy, vp.pic_amount, vp.nx, vp.ny, vp.npartcl)

    gridSize1D = convert(
        Int64,
        floor((vp.max_npartcl - vp.npartcl + blockSize1D - 1) / blockSize1D),
    )

    #processing all particles
    pTs = @view gp.pT[vp.npartcl+1:end]
    @cuda blocks = gridSize1D threads = blockSize1D init_particles_T(pTs, vp.T_magma, vp.max_npartcl - vp.npartcl)

    pPhs = @view gp.pPh[vp.npartcl+1:end]
    @cuda blocks = gridSize1D threads = blockSize1D init_particles_Ph(pPhs, 1, vp.max_npartcl - vp.npartcl)
    println(vp.max_npartcl)
    println(vp.npartcl)
    println(vp.max_nmarker)
    println(vp.nmarker)

    #initin temerature of all markers that will appear with magma Temperature
    if (markers_flag)
        #processing all markers
        gridSize1D = Int64(floor((vp.max_nmarker - vp.nmarker + blockSize1D - 1) / blockSize1D))
        mTs = @view gp.mT[vp.nmarker+1:end]
        @cuda blocks = gridSize1D threads = blockSize1D init_particles_T(mTs, vp.T_magma, vp.max_nmarker - vp.nmarker)
    end

    synchronize()

    vp.pic_amount = pic_amount_tmp

end

"""
Function to check melt fraction in a 2D model.

This function calculates the melt fraction based on given grid parameters and variable parameters.
It also performs volume counting and identifies the largest volume.

# Arguments:
- `gp::GridParams`: Grid parameters containing necessary grid information.
- `vp::VarParams`: Variable parameters containing simulation variables.
- `mf_rock_c::CuArray{Float64,1}`: CUDA array for melt fraction rock values (optional).

# Returns:
- `maxVol::Int32`: The volume of the largest identified region.
- `maxIdx::Int32`: The index of the largest identified region.
- `sumVol::Int32`: The total volume counted.

# Notes:
- This function uses CUDA for parallel computation if GPU is available.
- It counts volumes that are larger than a certain boundary defined by `Ly_eruption`.
"""
function d2dm_check_melt_fracton(gp::GridParams, vp::VarParams, mf_rock_c)
    @time begin
        nxl = vp.nxl
        nyl = vp.nyl

        blockSizel = (28, 32)
        gridSizel = (
            (vp.nxl + blockSizel[1] - 1) ÷ blockSizel[1],
            (vp.nyl + blockSizel[2] - 1) ÷ blockSizel[2],
        )


        #mf_rock_c = CuArray{Float64,1}(undef, vp.nx * vp.ny)
        #CUDA.device!(0)
        #copyto!(mf_rock_c, gp.T)

        #Усредняется mf
        @cuda blocks = gridSizel threads = blockSizel average!(gp.mfl, gp.T, gp.C, vp.nl, vp.nx, vp.ny, mf_rock_c)

        synchronize()

        ccl(gp.mfl, gp.L, vp.tsh, vp.nxl, vp.nyl)

        copyto!(gp.L_host, gp.L)

        volumes = Dict{Int32,Int32}(-1 => 0)

        #counting volumes
        for iy = 0:(nyl-1)
            #taking into account only volumes higher then certain boundary Ly_eruption
            if (iy * vp.dy * vp.nl < vp.Ly_eruption)
                continue
            end
            for ix = 1:vp.nxl
                if gp.L_host[iy*nxl+ix] >= 0
                    if haskey(volumes, gp.L_host[iy*nxl+ix])
                        volumes[gp.L_host[iy*nxl+ix]] =
                            volumes[gp.L_host[iy*nxl+ix]] + 1
                    else
                        volumes[gp.L_host[iy*nxl+ix]] = 0
                        volumes[gp.L_host[iy*nxl+ix]] =
                            volumes[gp.L_host[iy*nxl+ix]] + 1
                    end
                end
            end
        end

        #maxVol - numbrer of cells
        maxVol = -1
        maxIdx = -1

        sumVol = 0


        #searching for max vol
        for (idx, vol) in volumes
            sumVol = sumVol + vol
            if vol > maxVol
                maxVol = vol
                maxIdx = idx
            end
        end

        dxl = vp.dx * vp.nl
        dyl = vp.dy * vp.nl

        if (maxVol * dxl * dyl >= gp.critVol[vp.iSample])
            #println(volumes)
            #return -1,-1
        end

    end

    return maxVol, maxIdx, sumVol
end

"""
Advances the particles and markers based on the eruption location.

# Arguments
- `gp::GridParams`: Parameters related to the grid.
- `vp::VarParams`: Parameters related to variables.
- `maxVol`: Maximum volume of interest.
- `maxIdx`: Maximum index of interest.
- `it`: Current iteration.
- `markers_flag::Bool`: Flag indicating whether markers should be advected.

# Returns
- None

This function calculates the center of eruptions, updates the eruption positions,
advects particles and optionally markers based on the eruption location. It also
updates sample indices and marks that an eruption has occurred.
"""
function d2dm_eruption_advection(gp::GridParams, vp::VarParams, maxVol, maxIdx, it, markers_flag::Bool)
    @time begin

        cell_idx = CuArray{Int32,1}(undef, maxVol)
        cell_idx_host = Array{Int32,1}(undef, maxVol)

        dxl = vp.dx * vp.nl
        dyl = vp.dy * vp.nl

        next_idx = 0
        for idx = 0:(vp.nxl*vp.nyl)-1
            if gp.L_host[idx+1] == maxIdx
                next_idx = next_idx + 1
                cell_idx_host[next_idx] = idx
            end
        end

        dxl = vp.dx * vp.nl
        dyl = vp.dy * vp.nl

        #here i can find center of eruptions
        #i can find min/max of idx and find median
        cell_idx_x_max = 0.0
        cell_idx_x_min = vp.nxl * dxl
        cell_idx_y_max = 0.0
        cell_idx_y_min = vp.nyl * dyl
        for val in cell_idx_host
            if val != 0
                val_x = (val % vp.nxl) * dxl
                val_y = (val ÷ vp.nyl) * dyl

                cell_idx_x_max = max(val_x, cell_idx_x_max)
                cell_idx_x_min = min(val_x, cell_idx_x_min)
                cell_idx_y_max = max(val_y, cell_idx_y_max)
                cell_idx_y_min = min(val_y, cell_idx_y_min)
            end
        end

        println("Eruption center")
        println((cell_idx_x_max + cell_idx_x_min) / 2.0)
        println((cell_idx_y_max + cell_idx_y_min) / 2.0)

        append!(gp.erupt_x, (cell_idx_x_max + cell_idx_x_min) / 2.0)
        append!(gp.erupt_y, (cell_idx_y_max + cell_idx_y_min) / 2.0)



        copyto!(cell_idx, cell_idx_host)

        local blockSize1D = dev_thread
        local gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

        #advect particles
        @cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(gp.px, gp.py, cell_idx, vp.gamma, dxl, dyl, vp.npartcl, maxVol, vp.nxl, vp.nyl)
        synchronize()



        if (markers_flag)
            gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D
            #advect markers
            @cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(gp.mx, gp.my, cell_idx, vp.gamma, dxl, dyl, vp.nmarker, maxVol, vp.nxl, vp.nyl)
        end
        synchronize()

        vp.iSample = vp.iSample + 1

        vp.is_eruption = true
        append!(gp.eruptionSteps, it)
    end

end

"""
d2dm_inserting_dykes(gp::GridParams, vp::VarParams, it, markers_flag)

Inserts dykes into a simulation grid and updates particle and marker positions accordingly.

# Arguments
- `gp`: Grid parameters including dyke positions and properties.
- `vp`: Variable parameters including particle and marker details.
- `it`: Current iteration index indicating the current set of dykes to insert.
- `markers_flag`: A boolean flag indicating whether markers should also be updated.

# Returns
- Returns `-1` if the number of particles exceeds the maximum capacity, otherwise proceeds silently.

This function iterates over each dyke specified in `gp.ndykes[it]`, updates the particle and marker positions based on the dyke's intrusion, and adjusts the total number of particles and markers accordingly. If `markers_flag` is true, it also updates marker positions similarly to the particles.
"""
function d2dm_inserting_dykes(gp::GridParams, vp::VarParams, it, markers_flag)
    @time begin
        for i = 1:gp.ndykes[it]
            vp.idyke = vp.idyke + 1
            idyke = vp.idyke

            blockSize1D = 896
            gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

            @cuda blocks = gridSize1D threads = blockSize1D advect_particles_intrusion(
                gp.px,
                gp.py,
                gp.dyke_a[idyke],
                gp.dyke_b[idyke],
                gp.dyke_x[idyke],
                gp.dyke_y[idyke],
                gp.dyke_t[idyke],
                vp.nu,
                vp.G,
                gp.ndykes[it],
                vp.npartcl
            )

            dyke_start = gp.particle_edges[idyke]
            dyke_end = gp.particle_edges[idyke+1]
            np_dyke = dyke_end - dyke_start

            if (vp.npartcl + np_dyke > vp.max_npartcl)
                @printf("ERROR: number of particles exceeds maximum value, increase capacity\n")
                return -1
            end


            pxs = @view gp.px[(vp.npartcl+1):(vp.npartcl+np_dyke)]
            px_dykess = @view gp.px_dykes[(dyke_start+1):(dyke_start+np_dyke)]
            pys = @view gp.py[(vp.npartcl+1):(vp.npartcl+np_dyke)]
            py_dykess = @view gp.py_dykes[(dyke_start+1):(dyke_start+np_dyke)]


            copyto!(pxs, px_dykess)
            copyto!(pys, py_dykess)


            vp.npartcl += np_dyke

            gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D


            if (markers_flag)
                @cuda blocks = gridSize1D threads = blockSize1D advect_particles_intrusion(
                    gp.mx,
                    gp.my,
                    gp.dyke_a[idyke],
                    gp.dyke_b[idyke],
                    gp.dyke_x[idyke],
                    gp.dyke_y[idyke],
                    gp.dyke_t[idyke],
                    vp.nu,
                    vp.G,
                    gp.ndykes[it],
                    vp.nmarker,
                )

                synchronize()

                vp.nmarker += gp.marker_edges[idyke+1] - gp.marker_edges[idyke]
            end
        end
    end
end

"""
`d2dm_p2g_interpolation(gp::GridParams, vp::VarParams) -> Nothing`

Performs the particle to grid interpolation for a 2D model.

# Arguments:
- `gp`: A `GridParams` struct containing parameters related to the grid.
- `vp`: A `VarParams` struct containing variables like particle positions and dimensions.

# Returns:
- Nothing

This function initializes the temperature (`T`), concentration (`C`), and weights (`wts`) arrays in `gp`.
It then performs two CUDA kernel calls:
1. `p2g_project!` to project particles onto the grid.
2. `p2g_weight!` to calculate weights based on the projected values.

The function measures the execution time using `@time`.

# Notes:
- The grid and block sizes are dynamically calculated based on the number of particles and dimensions.
- Synchronization calls (`synchronize()`) ensure that all CUDA operations have completed before proceeding.
"""
function d2dm_p2g_interpolation(gp::GridParams, vp::VarParams)
    @time begin
        fill!(gp.T, 0)
        fill!(gp.C, 0)
        fill!(gp.wts, 0)

        blockSize1D = 896
        gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

        blockSize = (28, 32)
        gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

        @cuda blocks = gridSize1D threads = blockSize1D p2g_project!(gp.T, gp.C, gp.wts, gp.px, gp.py, gp.pT, gp.pPh, vp.dx, vp.dy, vp.nx, vp.ny, vp.npartcl, vp.npartcl0)
        synchronize()


        @cuda blocks = gridSize threads = blockSize p2g_weight!(gp.T, gp.C, gp.wts, vp.nx, vp.ny)
        synchronize()
    end
end

"""
    d2dm_particles_injection(gp::GridParams, vp::VarParams)

Inject particles into the simulation grid based on the given parameters.

# Arguments
- `gp`: A `GridParams` object containing parameters related to the grid.
- `vp`: A `VarParams` object containing various variables used in the simulation.

# Returns
- None

This function performs the following steps:
1. Determines the block and grid sizes for particle counting and injection.
2. Counts the number of particles in each grid cell using CUDA.
3. Synchronizes the GPU to ensure all operations are complete.
4. Injects additional particles where necessary based on the minimum count per cell.
5. Ensures the total number of particles does not exceed the maximum capacity.
6. Updates the number of particles if new particles were injected.

# Notes
- The function uses CUDA for parallel computation, making it suitable for large-scale simulations.
"""
function d2dm_particles_injection(gp::GridParams, vp::VarParams)
    @time begin
        blockSize1D = dev_thread
        gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

        blockSize = (28, 32)
        gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

        #@printf("%s particle injection	   | ", bar2)
        fill!(gp.pcnt, 0)

        #count particles
        @cuda blocks = gridSize1D threads = blockSize1D count_particles!(gp.pcnt, gp.px, gp.py, vp.dx, vp.dy, vp.nx, vp.ny, vp.npartcl)
        synchronize()

        gp.npartcl_h[1] = vp.npartcl
        copyto!(gp.npartcl_d, gp.npartcl_h)

        min_pcount = 2

        #inject particles where theit not enough
        @cuda blocks = gridSize threads = blockSize inject_particles!(gp.px, gp.py, gp.pT, gp.pPh, gp.npartcl_d, gp.pcnt, gp.T, gp.C, vp.dx, vp.dy, vp.nx, vp.ny, min_pcount, vp.max_npartcl)
        synchronize()

        copyto!(gp.npartcl_h, gp.npartcl_d)
        new_npartcl = gp.npartcl_h[1]

        if new_npartcl > vp.max_npartcl
            fprintf(
                stderr,
                "ERROR: number of particles exceeds maximum value, increase capacity\n",
            )
            exit(EXIT_FAILURE)
        end

        if (new_npartcl > vp.npartcl)
            @printf("(%03d) | ", new_npartcl - vp.npartcl)
            vp.npartcl = new_npartcl
        else
            @printf("(000) | ")
        end

    end
end
