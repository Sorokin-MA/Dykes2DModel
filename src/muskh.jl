Base.@kwdef mutable struct DykeParam
    a::Float64 = 4
    b::Float64 = 0.5
    x::Float64 = 0
    y::Float64 = 0
    phi::Float64 = 0
    P_in::Float64 = 1
end


function insert_dyke_gpu!(Sxx, Syy, Sxy, XX, YY, nx, ny,
    dyke_param_a::Float64, dyke_param_b::Float64, dyke_param_P_in::Float64,
    dyke_param_x::Float64, dyke_param_y::Float64, dyke_param_phi::Float64)

    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ((ix > nx) || (iy > ny))
        return nothing
    end

    cur_index = ix + nx * (iy - 1)

    m::Float64 = (dyke_param_a - dyke_param_b) / (dyke_param_a + dyke_param_b) #Variable in Joukovskiy equasion
    nu::Float64 = 0.3 #poussion coefficient
    eta::Float64 = (1 - 2 * nu) / (1 - nu) / 2
    rc::Float64 = 20 # rho_*
    Pcr::Float64 = dyke_param_P_in # Fluid pressure on cavity
    Po::Float64 = 3 # Fluid pressure on external boundary


    point_x = XX[ix] - dyke_param_x
    point_y = YY[iy] - dyke_param_y


    rho, upsilon = d2dm_cart_to_polar(point_x, point_y)

    #move to coordinate system where ellips aligned with x,y
    upsilon = upsilon + dyke_param_phi

    Z_real = rho * exp(1im * upsilon)

    #FIXME why 1/2? 
    R = 1 / 2

    #Reverse Zhoukovski
    if (real(Z_real) >= 0)
        Zr_rev_z = Z_real + sqrt(Z_real^2 - m)
    else
        Zr_rev_z = Z_real - sqrt(Z_real^2 - m)
    end

    X = real(Zr_rev_z)
    Y = imag(Zr_rev_z)

    Rho = rho
    alpha = upsilon

    rho, upsilon = d2dm_cart_to_polar(X, Y)

    #equasions solved in coordinate system before Zhoukovski transformation
    if (rho >= 1)
        Srr = (eta * rho^2 * Pcr * m^3 * cos(2 * upsilon) * log(1 / (rho^32)) + eta * rho^2 * Pcr * m^4 * log(1 / (rc^8)) + eta * Pcr * m^4 * log(rho^8 * rc^8) + eta * rho^6 * Pcr * log(rc^8) + eta * m^2 * rho^4 * Po * log(1 / (rho^32)) + eta * m^2 * rho^4 * Pcr * log(rho^32) + eta * m * rho^6 * Po * cos(2 * upsilon) * log(rho^32) + eta * m^2 * rho^2 * Pcr * log(1 / (rc^8)) + eta * rho^2 * Po * m^4 * log(rc^8) + eta * rho^8 * Pcr * log(1 / rc^8 * rho^8) + eta * rho^8 * Po * log(1 / rho^8 * rc^8) + 8 * eta * Po * m^4 + 24 * eta * m^2 * rho^2 * Pcr + 16 * eta * m^2 * rho^4 * Po - 16 * eta * m^2 * rho^4 * Pcr - 8 * eta * rho^6 * Pcr * m^2 + 8 * eta * rho^6 * Po * m^2 - 8 * eta * Pcr * m^4 - 24 * eta * m^2 * rho^2 * Po + eta * rho^6 * Pcr * m^2 * log(rc^8) + 8 * eta * rho^2 * Pcr * m^4 - 8 * eta * rho^2 * Po * m^4 + 8 * eta * rho^2 * Pcr * m^3 * cos(2 * upsilon) - 8 * eta * m^3 * Pcr * cos(2 * upsilon) + 8 * eta * m^3 * Po * cos(2 * upsilon) + eta * Po * m^4 * log(1 / (rho^8 * rc^8)) + eta * rho^6 * Po * log(1 / (rc^8)) + 24 * eta * m * rho^4 * Po * cos(2 * upsilon) - 8 * eta * m^2 * rho^2 * Po * cos(4 * upsilon) + 8 * eta * m^2 * rho^4 * Po * cos(4 * upsilon) - 24 * eta * m * rho^6 * Po * cos(2 * upsilon) + 8 * eta * m^2 * rho^2 * Pcr * cos(4 * upsilon) - 8 * eta * m^2 * rho^4 * Pcr * cos(4 * upsilon) - 8 * eta * rho^2 * Po * m^3 * cos(2 * upsilon) + eta * m^2 * rho^4 * Pcr * log(rho^16) * cos(4 * upsilon) + eta * m^2 * rho^4 * Po * log(1 / (rho^16)) * cos(4 * upsilon) - 24 * eta * m * rho^4 * Pcr * cos(2 * upsilon) + eta * rho^2 * Po * m^3 * cos(2 * upsilon) * log(rho^32) + 24 * eta * m * rho^6 * Pcr * cos(2 * upsilon) + eta * m^2 * rho^2 * Po * log(rc^8) + eta * rho^6 * Po * m^2 * log(1 / (rc^8)) + eta * m * rho^6 * Pcr * cos(2 * upsilon) * log(1 / (rho^32))) / (m^2 * rho^4 * cos(4 * upsilon) * log(rc^16) + m^2 * rho^4 * log(rc^32) + rho^6 * m * cos(2 * upsilon) * log(1 / (rc^32)) + rho^2 * m^3 * cos(2 * upsilon) * log(1 / (rc^32)) + rho^8 * log(rc^8) + m^4 * log(rc^8))
        Stt = (eta * Pcr * m^4 * log(rho^8 * rc^8) + eta * m^2 * rho^4 * Po * log(1 / (rho^32)) + eta * m^2 * rho^4 * Pcr * log(rho^32) + eta * m^2 * rho^2 * Pcr * log(rc^8) + eta * rho^6 * Po * m^2 * log(rc^8) + eta * rho^2 * Pcr * m^4 * log(rc^8) + eta * m^2 * rho^2 * Po * log(1 / (rc^8)) + eta * rho^8 * Pcr * log(1 / rc^8 * rho^8) + eta * rho^8 * Po * log(1 / rho^8 * rc^8) + 16 * eta * Po * m^4 - 24 * eta * m^2 * rho^2 * Pcr + 16 * eta * m^2 * rho^4 * Po - 16 * eta * m^2 * rho^4 * Pcr + 8 * eta * rho^6 * Pcr * m^2 - 8 * eta * rho^6 * Po * m^2 - 16 * eta * Pcr * m^4 + 24 * eta * m^2 * rho^2 * Po - 8 * eta * rho^2 * Pcr * m^4 + 8 * eta * rho^2 * Po * m^4 + 56 * eta * rho^2 * Pcr * m^3 * cos(2 * upsilon) + 8 * eta * m^3 * Pcr * cos(2 * upsilon) - 8 * eta * m^3 * Po * cos(2 * upsilon) + eta * Po * m^4 * log(1 / (rho^8 * rc^8)) - 24 * eta * m * rho^4 * Po * cos(2 * upsilon) + 8 * eta * m^2 * rho^2 * Po * cos(4 * upsilon) + 8 * eta * m^2 * rho^4 * Po * cos(4 * upsilon) + 24 * eta * m * rho^6 * Po * cos(2 * upsilon) - 8 * eta * m^2 * rho^2 * Pcr * cos(4 * upsilon) - 8 * eta * m^2 * rho^4 * Pcr * cos(4 * upsilon) - 56 * eta * rho^2 * Po * m^3 * cos(2 * upsilon) + eta * m^2 * rho^4 * Pcr * log(rho^16) * cos(4 * upsilon) + eta * m^2 * rho^4 * Po * log(1 / (rho^16)) * cos(4 * upsilon) + 24 * eta * m * rho^4 * Pcr * cos(2 * upsilon) - 24 * eta * m * rho^6 * Pcr * cos(2 * upsilon) + eta * rho^6 * Pcr * log(1 / (rc^8)) + eta * rho^6 * Pcr * m^2 * log(1 / (rc^8)) + eta * rho^2 * Po * m^3 * cos(2 * upsilon) * log(rho^32 * rc^32) + eta * rho^6 * Po * log(rc^8) + eta * rho^2 * Po * m^4 * log(1 / (rc^8)) + eta * m * rho^6 * Po * cos(2 * upsilon) * log(1 / rc^32 * rho^32) + 8 * eta * rho^8 * Pcr - 8 * eta * rho^8 * Po + eta * m * rho^6 * Pcr * cos(2 * upsilon) * log(1 / rho^32 * rc^32) + eta * rho^2 * Pcr * m^3 * cos(2 * upsilon) * log(1 / (rho^32 * rc^32))) / (m^2 * rho^4 * cos(4 * upsilon) * log(rc^16) + m^2 * rho^4 * log(rc^32) + rho^6 * m * cos(2 * upsilon) * log(1 / (rc^32)) + rho^2 * m^3 * cos(2 * upsilon) * log(1 / (rc^32)) + rho^8 * log(rc^8) + m^4 * log(rc^8))
        Srt = eta * m * sin(2 * upsilon) * (-2 * rho^6 * Pcr * log(rc) + 2 * rho^2 * log(rc) * Po * m^2 - 2 * rho^4 * log(rc) * Po * m^2 + 2 * rho^4 * Pcr * log(rc) * m^2 + 2 * m * Pcr * rho^2 * cos(2 * upsilon) - 2 * m * Po * rho^2 * cos(2 * upsilon) - 2 * rho^4 * Pcr * m^2 + 2 * rho^6 * log(rc) * Po + 2 * rho^4 * Po * m^2 - 2 * m * rho^4 * Pcr * cos(2 * upsilon) - m^2 * Pcr + m^2 * Po - 3 * rho^2 * Po * m^2 + 3 * rho^4 * Po - 3 * rho^4 * Pcr + 3 * rho^2 * Pcr * m^2 - 3 * rho^6 * Po + 3 * rho^6 * Pcr + 2 * m * rho^4 * Po * cos(2 * upsilon) + 2 * rho^4 * Pcr * log(rc) - 2 * rho^2 * Pcr * log(rc) * m^2 - 2 * rho^4 * log(rc) * Po) / log(rc) / (4 * m^2 * rho^4 * cos(2 * upsilon)^2 + 2 * m^2 * rho^4 - 4 * rho^6 * m * cos(2 * upsilon) - 4 * rho^2 * m^3 * cos(2 * upsilon) + rho^8 + m^4)

        # Sxx[cur_index] = Sxx[cur_index] + 1/2*(((-2*Rho.^2+1+Rho.^4).*Srr+(-2*Rho.^2-1-Rho.^4).*Stt).*cos(2*alpha)+(-2*Rho.^2+1+Rho.^4).*Srr+(2*Rho.^2+1+Rho.^4).*Stt+(-2*Rho.^4*sin(2*alpha)+2*sin(2*alpha)).*Srt)./(-2*Rho.^2*cos(2*alpha)+Rho.^4+1);
        # Syy[cur_index] = Syy[cur_index] - 1/2*(((2*Rho.^2+1+Rho.^4).*Srr+(-Rho.^4+2*Rho.^2-1).*Stt).*cos(2*alpha)+(-2*Rho.^2-1-Rho.^4).*Srr+(-Rho.^4+2*Rho.^2-1).*Stt+(-2*Rho.^4*sin(2*alpha)+2*sin(2*alpha)).*Srt)./(-2*Rho.^2*cos(2*alpha)+Rho.^4+1);
        # Sxy[cur_index] = Sxy[cur_index] + 1/2*((2+2*Rho.^4).*Srt.*cos(2*alpha)+(-sin(2*alpha)+Rho.^4*sin(2*alpha)).*Srr+(sin(2*alpha)-Rho.^4*sin(2*alpha)).*Stt-4*Srt.*Rho.^2)./(-2*Rho.^2*cos(2*alpha)+Rho.^4+1)

        #dSxx = 
        #CUDA.atomic_add!(pointer(Sxx, idc(ix1, iy1, nx)), k11 * pT[ip])

        Sxx[cur_index] = Sxx[cur_index] + 1 / 2 * (((-2 * Rho .^ 2 + 1 + Rho .^ 4) .* Srr + (-2 * Rho .^ 2 - 1 - Rho .^ 4) .* Stt) .* cos(2 * alpha) + (-2 * Rho .^ 2 + 1 + Rho .^ 4) .* Srr + (2 * Rho .^ 2 + 1 + Rho .^ 4) .* Stt + (-2 * Rho .^ 4 * sin(2 * alpha) + 2 * sin(2 * alpha)) .* Srt) ./ (-2 * Rho .^ 2 * cos(2 * alpha) + Rho .^ 4 + 1)
        Syy[cur_index] = Syy[cur_index] - 1 / 2 * (((2 * Rho .^ 2 + 1 + Rho .^ 4) .* Srr + (-Rho .^ 4 + 2 * Rho .^ 2 - 1) .* Stt) .* cos(2 * alpha) + (-2 * Rho .^ 2 - 1 - Rho .^ 4) .* Srr + (-Rho .^ 4 + 2 * Rho .^ 2 - 1) .* Stt + (-2 * Rho .^ 4 * sin(2 * alpha) + 2 * sin(2 * alpha)) .* Srt) ./ (-2 * Rho .^ 2 * cos(2 * alpha) + Rho .^ 4 + 1)
        Sxy[cur_index] = Sxy[cur_index] + 1 / 2 * ((2 + 2 * Rho .^ 4) .* Srt .* cos(2 * alpha) + (-sin(2 * alpha) + Rho .^ 4 * sin(2 * alpha)) .* Srr + (sin(2 * alpha) - Rho .^ 4 * sin(2 * alpha)) .* Stt - 4 * Srt .* Rho .^ 2) ./ (-2 * Rho .^ 2 * cos(2 * alpha) + Rho .^ 4 + 1)
    else
        Sxx[cur_index] = Sxx[cur_index] + 1
        Syy[cur_index] = Syy[cur_index] + 1
        Sxy[cur_index] = Sxy[cur_index] + 1
    end

    return nothing
end

function d2dm_polar_to_cart(r, phi)
    return r .* cos.(phi), r .* sin.(phi)
end

#TODO
#boilerplate
function d2dm_cart_to_polar(x::Vector, y::Vector)
    tmp = y ./ x
    r, phi = sqrt.(x .^ 2 .+ y .^ 2), atan.(tmp)
    for i in eachindex(phi)
        if (x[i] < 0)
            phi[i] += pi
        end
    end
    return r, phi
end

function d2dm_cart_to_polar(x::Float64, y::Float64)
    tmp = y / x
    r, phi = sqrt(x^2 + y^2), atan(tmp)
    if (x < 0)
        phi += pi
    end
    return r, phi
end


```
Get ints on peremeter of ellipsis
```
function get_points!(dyke_param, num_of_points)
    upsilon = 0:(2*pi)/(num_of_points-1):2*pi
    rho = 1

    Z_real = rho .* exp.(1im * upsilon)

    R = 1 / 2
    m::Float64 = (dyke_param.a - dyke_param.b) / (dyke_param.a + dyke_param.b) #Variable in Joukovskiy equasion

    #Zhoukovski transformation
    Z_new = R .* (Z_real .+ m ./ Z_real)

    x, y = real(Z_new), imag(Z_new)
    r, phi = d2dm_cart_to_polar(x, y)

    phi = phi .- dyke_param.phi

    x, y = d2dm_polar_to_cart(r, phi)

    x = x .+ dyke_param.x
    y = y .+ dyke_param.y

    return x, y, r, phi .+ dyke_param.phi
end

#Averaging grid to particle
function d2dm_blerp(x1, x2, y1, y2, f11, f12, f21, f22, x, y)
    invDxDy = 1.0 / ((x2 - x1) * (y2 - y1))

    dx1 = x - x1
    dx2 = x2 - x

    dy1 = y - y1
    dy2 = y2 - y

    return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1)
end


```
Function interpolate point with blerp, interpolate Sxx, Sxy, Syy accordingly
and then calculating maximal sigma
```
function get_sigma2(x, y, XX, YY, Sxx, Syy, Sxy)
    x_lf = 0.0
    y_lf = 0.0


    for i in eachindex(collect(XX))
        if (XX[i] > x)
            x_lf = i - 1
            break
        end
    end

    for i in eachindex(collect(YY))
        if (YY[i] > y)
            y_lf = i - 1
            break
        end
    end

    small_dif_x = 0
    small_dif_y = 0


    Sxx_local = d2dm_blerp(XX[x_lf], XX[x_lf+1], YY[y_lf], YY[y_lf+1], Sxx[x_lf, y_lf], Sxx[x_lf, y_lf+1], Sxx[x_lf+1, y_lf], Sxx[x_lf+1, y_lf+1], x + small_dif_x, y + small_dif_y)
    Sxy_local = d2dm_blerp(XX[x_lf], XX[x_lf+1], YY[y_lf], YY[y_lf+1], Sxy[x_lf, y_lf], Sxy[x_lf, y_lf+1], Sxy[x_lf+1, y_lf], Sxy[x_lf+1, y_lf+1], x + small_dif_x, y + small_dif_y)
    Syy_local = d2dm_blerp(XX[x_lf], XX[x_lf+1], YY[y_lf], YY[y_lf+1], Syy[x_lf, y_lf], Syy[x_lf, y_lf+1], Syy[x_lf+1, y_lf], Syy[x_lf+1, y_lf+1], x + small_dif_x, y + small_dif_y)

    mat_local = [Sxx_local Sxy_local; Sxy_local Syy_local]

    l_vecs = eigvecs(mat_local)
    l_vals = eigvals(mat_local)

    #TODO:check if sorted right
    return l_vals[2], l_vecs
end


function calc_cent_of_next_dyke(dyke_param, Sxx, Syy, Sxy, XX, YY, y_limit, Y_right_lim, X_lim)

    num_of_points_on_ellipsis::Int = 1134
    xpoints, ypoints, point_dist, point_angle = get_points!(dyke_param, num_of_points_on_ellipsis)
    l_vecs = Matrix{Float64}(undef, 2, 2)

    for i in eachindex(ypoints)
		x_new = X_lim/4 + X_lim/2 * rand()
        y_new = 3.0
        #println(ypoints)
        if ypoints[i] > y_limit
            println("REACHED SURFACE!")
            #_, l_vecs = get_sigma2(x_new, y_new, XX, YY, Sxx, Syy, Sxy)
            tmp, l_vecs = get_sigma2(xpoints[1], ypoints[1], XX, YY, Sxx, Syy, Sxy)
            return x_new, y_new, xpoints, ypoints, l_vecs
        end
		if (xpoints[i] >= X_lim) || (xpoints[i] <= 0)
            println("REACHED BOUNDARY!")
            #_, l_vecs = get_sigma2(x_new, y_new, XX, YY, Sxx, Syy, Sxy)
            tmp, l_vecs = get_sigma2(xpoints[1], ypoints[1], XX, YY, Sxx, Syy, Sxy)
            return x_new, y_new, xpoints, ypoints, l_vecs
        end

    end

    #println(xpoints)
    #2. calculate preassure on boundary
    #2.1. find normal to ellipsis
    #2.2. find average preassure tensor in points
    #2.3. find pressure in point by multplying
    #3. calculate delta y on boundaries
    #4. Kalculate K
    #5. choose point based on K

    #find sigmas in observation points
    sigma2_in_points = Array{Float64}(undef, 0)

    for i in eachindex(xpoints)
        sigma2, _ = get_sigma2(xpoints[i], ypoints[i], XX, YY, Sxx, Syy, Sxy)
        append!(sigma2_in_points, sigma2)
    end

    #println(size(sigma2_in_points)[1])
    #println(sigma2_in_points)

    c = dyke_param.a + dyke_param.b
    K = Array{Float64}(undef, 0)
    K_size::Int64 = Int(size(sigma2_in_points)[1])
    g = 9.8
    rho_m = 2800


    for i in eachindex(sigma2_in_points)
        i_opposite = Int((i + K_size ÷ 2) % (K_size) + 1)
        sigma_dif = sigma2_in_points[i] - sigma2_in_points[i_opposite]

        #NOTE pseudo penny shaped
        #NOTE here - ypoints - from surface, i guess...
        delta_y = (sigma_dif) / (2 * c) - rho_m * g * ((Y_right_lim - ypoints[i]) - (Y_right_lim - ypoints[i_opposite])) / (2 * c)
        append!(K, 4 / 3 * pi * delta_y * c * sqrt(pi * c))

        #NOTE: eliptical
		#       c = 2*point_dist[i]
		#       phi = point_angle[i]
		#       delta_y = (sigma_dif)/(2*c) - rho_m * g *((Y_right_lim-ypoints[i]) - (Y_right_lim-ypoints[i_opposite]))/(2*c)
		# C_00 = ypoints[i];
		# append!(K,  C_00/(dyke_param.a*dyke_param.b)*(pi/(dyke_param.a*dyke_param.b))^(1/2)*(dyke_param.a^2*sin(phi)^2 + dyke_param.b^2*cos(phi)^2)^(1.0/4))


        #println("sigma diff - $tmp_sigma_dif")
    end


    _, indx = findmax(K)

    # println(K)
    # println(size(K)[1])
    # println("indx - $indx")

    _, l_vecs = get_sigma2(xpoints[indx], ypoints[indx], XX, YY, Sxx, Syy, Sxy)

    return xpoints[indx], ypoints[indx], xpoints, ypoints, l_vecs
end

