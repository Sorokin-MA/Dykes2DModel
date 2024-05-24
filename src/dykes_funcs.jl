bar1 = "\n├──"
bar2 = "\n\t ├──"

"""
Function to count 1D index based on 2D indexes.

NOTE:
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
function idc(ix, iy, nx)
	return ((iy)*nx + ix+1)
end


#Functon to write number of CUDA device
function kernel()
	dev = Ref{Cint}()
	CUDA.cudaGetDevice(dev)
	@cuprint("Running on device $(dev[])\n")
	return
end


#Basic info aobut GPU
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

#helper function which helps to read from IO stream
function read_par(par, ipar)
	par_name_2 = par[ipar]
	ipar_2 = ipar + 1
	return par_name_2, ipar_2
end

#Averaging grid to particle
function blerp(x1, x2, y1, y2, f11, f12, f21, f22, x, y)
	invDxDy = 1.0 / ((x2 - x1) * (y2 - y1))

	dx1 = x - x1
	dx2 = x2 - x

	dy1 = y - y1
	dy2 = y2 - y

	return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1)
end


function dmf_rhyolite(T)
	t1 = T * T
	t9 = exp(0.961026e3 - 0.186618e-5 * t1 * T + t1 * 0.447948e-2 + T * (-0.359050e1))
	t12 = (0.1e1 + t9) * (0.1e1 + t9)
	return 0.559856e-5 / t12 * t9 * (t1 - 0.160022e4 * T + 0.641326e6)
end


function dmf_basalt(T)
	t1 = T * T
	t11 = exp(0.143636887899999948e3 - 0.2214446257e-6 * t1 * T + t1 * 0.572468110399999928e-3 + T * (-0.494427718499999891e0))
	t14 = (0.1e1 + t11) ^ 0.2e1
	return 0.6643338771e-6 * (t1 - 0.1723434948e4 * T + 0.7442458310e6) * t11 / t14
end


function mf_rhyolite(T)
	t2 = T * T;
	t7 = exp(0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 - 0.1866187556e-5 * t2 * T);
	return 0.1e1 / (0.1e1 + t7);
end


function mf_basalt(T)
	t2 = T * T;
	t7 = exp(960 - 3.554 * T + 0.4468e-2 * t2 - 1.907e-06 * t2 * T);
	return 0.1e1 / (0.1e1 + t7);
end


#coefficient which involved in heat equasion
function dmf_magma(T)
	return dmf_rhyolite(T)
end

#coefficient which involved in heat equasion
function dmf_rock(T)
	return dmf_basalt(T)
end

#melt fraction of magma
function mf_magma(T)
	return mf_rhyolite(T)
end

#melt fraction of host rocks
function mf_rock(T)
	return mf_basalt(T)
end

"""
	update_T!(T,  T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny)

Solve heat equasion
"""
function update_T!(T,  T_old, T_top, T_bot, C, lam_r_rhoCp, lam_m_rhoCp, L_Cp, dx, dy, dt, nx, ny)
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x - 1
	iy = (blockIdx().y-1) * blockDim().y + threadIdx().y - 1

	if (ix > nx-1) || (iy > (ny-1))
		return
	end

	qxw::Float64, qxe::Float64, qys::Float64, qyn::Float64 = 0.0, 0.0, 0.0, 0.0

	if ix == 0
		qxw = 0.0
	else
		qxw = -(T[idc(ix,iy,nx)] - T[idc(ix-1,iy,nx)]) / dx
	end

	if ix == nx-1
		qxe = 0.0
	else
		qxe = -(T[idc(ix+1,iy,nx)] - T[idc(ix,iy,nx)]) / dx
	end


	if iy == 0
		qys = -2.0 * (T[idc(ix,iy,nx)] - T_bot) / dy
	else
		qys = -(T[idc(ix,iy,nx)] - T[idc(ix,iy-1,nx)]) / dy
	end


	if iy == (ny-1)
		qyn = -2.0 * (T_top - T[idc(ix,iy,nx)]) / dy
	else
		qyn = -(T[idc(ix,iy+1,nx)] - T[idc(ix,iy,nx)]) / dy
	end

	dmf::Float64 = dmf_magma(T[idc(ix,iy,nx)]) * C[idc(ix,iy,nx)] + dmf_rock(T[idc(ix,iy,nx)]) * (1.0 - C[idc(ix,iy,nx)])
	lam_rhoCp::Float64 = (lam_m_rhoCp * C[idc(ix,iy,nx)]) + lam_r_rhoCp * (1.0 - C[idc(ix,iy,nx)])

	chi::Float64 = lam_rhoCp / (1.0 + L_Cp * dmf)

	T[idc(ix,iy,nx)] += -dt * chi * ((qxe - qxw) / dx + (qyn - qys) / dy)
	return
end


"""
	init_particles_Ph(pPh, ph, npartcl)

initing particles out of defined particles as magma particles

# Arguments
- `pPh`: ? 
- `ph`: ?
- `npartcl`: ?
"""
function init_particles_Ph(pPh, ph, npartcl)
	ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	if ip > npartcl
		return
	end

	pPh[ip] = ph;
	return
end


#WARN:unneccesary function?
function sign(val)
	return (val > zero(val)) - (val < zero(val))
end


#TODO:specify description, wht if 'f'?
"""

	cart2ellipt(f, x, y)

Convert cartesian coordinates to elliptical

"""
function cart2ellipt(f, x, y)
	xi_eta_1 = acosh(max(0.5 / f * (sqrt((x + f) * (x + f) + y * y) + sqrt((x - f) * (x - f) + y * y)), 1.0))
	xi_eta_2 = acos(min(max(x / (f * cosh(xi_eta_1)), -1.0), 1.0)) * sign(y)
	return xi_eta_1, xi_eta_2
end


#TODO:specify description
"""

	rot2d(x, y, sb, cb)	

HZ

"""
function rot2d(x, y, sb, cb)
	return  x * cb - y * sb, x * sb + y * cb
end

#TODO:specify description
"""

	crack_params(a, b, nu, G)

HZ

"""
function crack_params(a, b, nu, G)

	f::Float64 = 2 * nu * (a + b) - 2 * a - b

	return (-2 * b * G / f, 0.5 * f / (nu - 1))
end

#TODO:specify description
"""
	disp_inf_stress(s, st, ct, c, nu, G, shxi, chxi, seta, ceta)	

HZ

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
	return u_v[1],u_v[2]
end


#TODO:specify description
"""
	displacements(st, ct, p, s1, s3, f, x, y, nu, G	)

Functions which displacements

"""
function displacements(st, ct, p, s1, s3, f, x, y, nu, G)
	x_y_1, x_y_2 = rot2d(x, y, -st, ct)
	if abs(x_y_1) < 1e-10
		x_y_1 = 1e-10
	end
	if abs(x_y_2) < 1e-10
		x_y_2 = 1e-10
	end


	xi_eta_1, xi_eta_2  = cart2ellipt(f, x_y_1, x_y_2)
	seta = sin(xi_eta_2)
	ceta = cos(xi_eta_2)
	shxi = sinh(xi_eta_1)
	chxi = cosh(xi_eta_1)
	u_v1_1, u_v1_2  = disp_inf_stress(s1 - p, st, ct, f, nu, G, shxi, chxi, seta, ceta)
	u_v2_1, u_v2_2 = disp_inf_stress(s3 - p, ct, -st, f, nu, G, shxi, chxi, seta, ceta)
	I = shxi * seta
	J = chxi * ceta
	u3 = 0.25 * p * f / G * (J * (3.0 - 4.0 * nu) - J)
	v3 = 0.25 * p * f / G * (I * (3.0 - 4.0 * nu) - I)

	u_v_1, u_v_2 = rot2d(u_v1_1 + u_v2_1 + u3, u_v1_2 + u_v2_2 + v3, st, ct)
	
	return -u_v_1, -u_v_2
end

"""
	advect_particles_intrusion(px, py, a, b, x, y, theta, nu, G, ndikes, npartcl)

Functions which calculate advection when particles intruded
"""
function advect_particles_intrusion(px, py, a, b, x, y, theta, nu, G, ndikes, npartcl)
	ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	if ip > npartcl
		return
	end

	p_a0_1, p_a0_2 = crack_params(a, b, nu, G)
	st = sin(theta)
	ct = cos(theta)
	u_v_1, u_v_2  = displacements(st, ct, p_a0_1, 0, 0, p_a0_2, px[ip] - x, py[ip] - y, nu, G)
	px[ip] += u_v_1
	py[ip] += u_v_2

	return nothing
end


#TODO:specify description
"""
	p2g_weight!(T, C, wts, nx, ny)
HZ
"""
function p2g_weight!(T, C, wts, nx, ny)
	ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
	iy = (blockIdx().y-1) * blockDim().y + threadIdx().y-1
	
	if (ix > nx) || (iy > ny-1)
		return
	end
	
	if wts[iy * nx + ix] == 0.0
		return
	end
	
	T[iy * nx + ix] = T[iy * nx + ix] / wts[iy * nx + ix]
	C[iy * nx + ix] = C[iy * nx + ix] / wts[iy * nx + ix]

	return nothing
end


#TODO:specify description
"""
	p2g_project!(T, C, wts, px, py, pT, pPh, dx, dy, nx, ny, npartcl, npartcl0)
HZ
"""
function p2g_project!(T, C, wts, px, py, pT, pPh, dx, dy, nx, ny, npartcl, npartcl0)
	ip = (blockIdx().x-1) * blockDim().x + threadIdx().x
	
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
	

	CUDA.atomic_add!(pointer(T, idc(ix1, iy1, nx)),k11 * pT[ip])
	CUDA.atomic_add!(pointer(T, idc(ix1, iy2, nx)),k12 * pT[ip])
	CUDA.atomic_add!(pointer(T, idc(ix2, iy1, nx)),k21 * pT[ip])
	CUDA.atomic_add!(pointer(T, idc(ix2, iy2, nx)),k22 * pT[ip])


	
	pC = (pPh == nothing) ? ((ip-1) > npartcl0 ? 1.0 : 0.0) : Float64(pPh[ip])
	

	CUDA.atomic_add!(pointer(C, idc(ix1, iy1, nx)),k11 * pC)
	CUDA.atomic_add!(pointer(C, idc(ix1, iy2, nx)),k12 * pC)
	CUDA.atomic_add!(pointer(C, idc(ix2, iy1, nx)),k21 * pC)
	CUDA.atomic_add!(pointer(C, idc(ix2, iy2, nx)),k22 * pC)


	CUDA.atomic_add!(pointer(wts, idc(ix1, iy1, nx)), k11)
	CUDA.atomic_add!(pointer(wts, idc(ix1, iy2, nx)), k12)
	CUDA.atomic_add!(pointer(wts, idc(ix2, iy1, nx)), k21)
	CUDA.atomic_add!(pointer(wts, idc(ix2, iy2, nx)), k22)

	return nothing
end


"""
	g2p!(T, T_old, px, py, pT, dx, dy, pic_amount, nx, ny, npartcl)

Grid to particles interpolation
"""
function g2p!(T, T_old, px, py, pT, dx, dy, pic_amount, nx, ny, npartcl)
	ip = (blockIdx().x-1) * blockDim().x + threadIdx().x
	
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

This function assigning unque lables to each cell if mf value more then trashhold
"""
function assignUniqueLables(mf, L, tsh, nx, ny)
	ix = (blockIdx().x-1)* blockDim().x + threadIdx().x-1
	iy = (blockIdx().y-1)* blockDim().y + threadIdx().y-1
	
	if (ix > nx-1  || iy > ny-1)
		return
	end
	
	if mf[(iy) * nx + ix + 1] >= tsh
		L[(iy) * nx + ix + 1] = (iy) * nx + ix
	else
		L[(iy)  * nx + ix + 1] = -1
	end
	return nothing
end


#TODO: describe function
"""
	cwLabel(L, nx, ny)	

I have no idea what this function do.
"""
function cwLabel(L, nx, ny)
	iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	
	if iy > ny
		return
	end
	
	for ix = nx - 1:-1:1
		if L[(iy-1) * nx + ix] >= 1 && L[(iy-1) * nx + ix + 1] >= 1
			L[(iy-1) * nx + ix] = L[(iy-1)* nx + ix + 1]
		end
	end
end

#TODO: describe function
"""
	find_root(L, idx)

I have no idea what this function do.
"""
function find_root(L, idx)
	label::Int32 = idx
	while L[label] != (label-1)
		label = L[label]+1
	end
	return label-1
end

#TODO: describe function
"""
	merge_labels!(L, div, nx::Int64, ny)

I have no idea what this function do.
"""
function merge_labels!(L, div, nx, ny)
	iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
	iy = Int64(floor(div / 2)) + iy * div - 1
	if iy > (ny - 2)
		return
	end

	for ix = 0:(nx-1)
		if (L[iy * nx + ix + 1] >= 0) && (L[(iy + 1) * nx + ix + 1] >= 0)
			lroot::Int32 = find_root(L, iy * nx + ix + 1)
			rroot::Int32 = find_root(L, (iy + 1) * nx + ix + 1)
			L[min(lroot, rroot)+1] = L[max(lroot, rroot)+1]
		end
	end

	return nothing
end


#TODO: describe function
"""
	relabel(L, nx, ny)

I have no idea what this function do.
"""
function relabel(L, nx, ny)
	ix = (blockIdx().x-1)* blockDim().x + threadIdx().x-1
	iy = (blockIdx().y-1)* blockDim().y + threadIdx().y-1

	if (ix > nx-1)  || (iy > ny-1)
		return
	end

	if L[iy * nx + ix + 1] >= 0
		L[iy * nx + ix + 1] = find_root(L, iy * nx + ix + 1)
	end

	return nothing
end


"""

	advect_particles_eruption(px, py, idx, gamma, dxl, dyl, npartcl, ncells, nxl, nyl)

Advect particles
"""
function advect_particles_eruption(px, py, idx, gamma, dxl, dyl, npartcl, ncells, nxl, nyl)
	ip = (blockIdx().x - 1 ) * blockDim().x + threadIdx().x-1

	if ip > npartcl-1
		return
	end

	u = 0.0
	v = 0.0

	for i = 0:ncells-1
		ic = idx[i+1]
		icx = ic % nxl
		icy = ic ÷ nxl

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
function average!(mfl, T, C, nl, nx, ny)
	ixl = (blockIdx().x-1) * blockDim().x + threadIdx().x-1
	iyl = (blockIdx().y-1) * blockDim().y + threadIdx().y-1

	if ixl > (nx ÷ nl - 1) || iyl > (ny ÷ nl - 1)
		return
	end

	avg = 0.0
	for ix = (ixl * nl):((ixl + 1) * nl - 1)
		if ix > nx - 1
			break
		end
		for iy = (iyl * nl):((iyl + 1) * nl - 1)
			if iy > ny - 1
				break
			end
			vf = C[iy * nx + ix + 1]
			avg = avg + (mf_magma(T[(iy * nx + ix + 1)])) * vf + mf_rock(T[(iy * nx + ix + 1)]) * (1 - vf)
		end
	end
	avg /= (nl * nl)
	mfl[iyl * (nx ÷ nl) + ixl + 1] = avg
	return nothing;
end


"""
	count_particles!(pcnt, px, py, dx, dy, nx, ny, npartcl)

Count particles in each cell
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
	inject_particles!(px, py, pT, pPh, npartcl, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl)

Function which inject particles
"""
function inject_particles!(px, py, pT, pPh, npartcl, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl)
	ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
	iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y - 1

	if ix > nx - 2 || iy > ny - 2
		return
	end

	if pcnt[iy * nx + ix + 1] < min_pcount
		for ioy = 0:1
			for iox = 0:1
				inx = ix + iox
				iny = iy + ioy
				new_npartcl = CUDA.atomic_add!(pointer(npartcl, 1), Int32(1))
				if new_npartcl > max_npartcl - 1
					break
				end
				ip = new_npartcl+1
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
	ccl(mf, L, tsh, nx, ny)

Function related with lables to
# Arguments
- `mf::CuArray`: Melt fraction grid.
- `L`: Helper grid.
- `tsh`: threashhold which describes which cells with which melt fraction to take into account.
- `nx`: x-axis grid resolution, [1]
- `ny`: y-axis grid resolution, [1]
"""
function ccl(mf, L, tsh, nx, ny)
	blockSize2D = (16, 32)
	gridSize2D = ((nx + blockSize2D[1] - 1) ÷ blockSize2D[1], (ny + blockSize2D[2] - 1) ÷ blockSize2D[2])
	
	#add lables to the points where ms > tsh
	CUDA.@sync begin
		@cuda blocks = gridSize2D threads = blockSize2D assignUniqueLables(mf, L, tsh, nx, ny)
	end
	blockSize1D = 32
	gridSize1D = (ny + blockSize1D - 1) ÷ blockSize1D

	#decreasing upcoming variables???
	CUDA.@sync begin
		@cuda  blocks = gridSize1D threads = blockSize1D cwLabel(L, nx, ny)
	end


	div = 2
	npw = Int64(ceil(log2(ny)))
	nyw = 1 << npw
	for i = 0:(npw-1)
		gridSize1D = Int64(floor(max((nyw + blockSize1D - 1) / blockSize1D / div, 1)))

		CUDA.@sync begin
			@cuda blocks = gridSize1D threads = blockSize1D merge_labels!(L, div, nx, ny)
		end

		div = div * 2
	end

	CUDA.@sync begin
		@cuda blocks = gridSize2D threads = blockSize2D relabel(L, nx, ny)
	end

	return nothing
end


"""
	init_particles_T(pT, T_magma, npartcl)

initing particles out of defined particles as magma particles

# Arguments
- `pT`: Temperature of particles, [°C]
- `T_magma`: Temperature of intruding magma, [°C]
- `npartcl`: maximal number of particles
"""
function init_particles_T(pT, T_magma, npartcl)
	ip = (blockIdx().x-1) * blockDim().x + threadIdx().x

	if (ip > (npartcl))
		return
	end

	pT[ip] = T_magma
	return
end


"""
	mf_magma(T)
	
# Arguments
- `T`: Temperature variable, [°C]
"""
function mf_magma(T)
	t2 = T * T
	t7 = exp(
		0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 -
		0.1866187556e-5 * t2 * T,
	)
	return 0.1e1 / (0.1e1 + t7)
end

#=
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
=#

"""
    write_h5(filename, data)

write data to HDF5 file

# Arguments
   - `filename`: the number of elements to compute.
   - `data`: the dimensions along which to perform the computation.
"""
function write_h5(filename,data)
    file = joinpath(dir, "$(filename)")    
    open(file, "w") do fid
         write(fid, data)
    end
end

"""

	small_mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl,max_nmarker, px,py,mx,my,h_px_dikes,pcnt, mfl)

write some variables in 'filename' in h5 format
"""
function small_mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl,max_nmarker, px,py,mx,my,h_px_dikes,pcnt, mfl);
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
		
		h_T = Array{Float64,1}(undef, nx*ny)			#array of double values from matlab script
		h_C = Array{Float64,1}(undef, nx*ny)			#array of double values from matlab script
		
		copyto!(h_T, T)
		copyto!(h_C, C)

		write(fid, "T", h_T)
		write(fid, "C", h_C)

		close(fid)
	end
end

"""

	mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl,max_nmarker, px,py,mx,my,h_px_dikes,pcnt, mfl)

write all variables in 'filename' in h5 format
"""
function mailbox_out(filename,T,pT, C, mT, staging,is_eruption,L,nx,ny,nxl,nyl,max_npartcl,max_nmarker, px,py,mx,my,h_px_dikes,pcnt, mfl);
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
		
		h_pcnt = Array{Int32,1}(undef, nx*ny)			#array of double values from matlab script
		h_T = Array{Float64,1}(undef, nx*ny)			#array of double values from matlab script
		h_C = Array{Float64,1}(undef, nx*ny)			#array of double values from matlab script
		h_pT = Array{Float64,1}(undef, max_npartcl)		#array of double values from matlab script
		h_mT = Array{Float64,1}(undef, max_nmarker)		#array of double values from matlab script
		h_L = Array{Int32,1}(undef, nxl*nyl)			#array of double values from matlab script
		h_px = Array{Float64,1}(undef, max_npartcl)			#array of double values from matlab script
		h_py = Array{Float64,1}(undef, max_npartcl)			#array of double values from matlab script
		h_mx = Array{Float64,1}(undef, max_nmarker)			#array of double values from matlab script
		h_my = Array{Float64,1}(undef, max_nmarker)			#array of double values from matlab script
		h_mfl = Array{Float64,1}(undef, nxl*nyl)			#array of double values from matlab script

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
		write(fid, "px_dikes", h_px_dikes)
		write(fid, "mfl", h_mfl)
		#write(fid, "L", h_L)

		copyto!(h_L, L)
		write(fid, "L", h_L)

		close(fid)
	end
end

function rand_limited(u, d)
	ans::Float64 = -1
	while((ans <=0) || (ans >=1))
		ans = rand(Normal(u, d),1)[1]
	end

	return ans
end

function rand_limited_2(u, d)
	ans::Float64 = -1
	while((ans <=0) || (ans >=1))
		ans = rand(Normal(u, d),1)[1]
	end

	return ans
end

function read_params(gp::GridParams, vp::VarParams)

    dpa = Array{Float64,1}(undef, 19)#array of double values from matlab script
    ipa = Array{Int32,1}(undef, 12)#array of int values from matlab script

    io = open(data_folder*"pa.bin", "r")
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
    gp.ndikes = Array{Int32,1}(undef, vp.nt)#number of dykes intruded on n-th time step
    read!(io, gp.ndikes)

    ndikes_all = 0

    #count all dykes
    for istep in 1:vp.nt
        ndikes_all = ndikes_all + gp.ndikes[istep]
    end

    #array which describes amount of particles in new dyke
    gp.particle_edges = Array{Int32,1}(undef, ndikes_all + 1)
    read!(io, gp.particle_edges)

    gp.marker_edges = Array{Int32,1}(undef, ndikes_all + 1)
    read!(io, gp.marker_edges)

    close(io)

    cap_frac = 1.5  #value to spcify how much particles we allow to inject in runtime
    vp.npartcl0 = vp.npartcl #initial amount of particles
    vp.max_npartcl = convert(Int64, vp.npartcl * cap_frac) + gp.particle_edges[ndikes_all+1] #???#count max particles
	println(vp.npartcl)
	println(gp.particle_edges[ndikes_all+1])
	println(vp.max_npartcl)
    nmarker0 = vp.nmarker

    vp.max_nmarker = vp.nmarker + gp.marker_edges[ndikes_all+1]


    #blockSize(16, 32);
    #gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);


    blockSize = (16, 32)
    gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

    gp.T = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.T_old = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.C = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.wts = CuArray{Float64,1}(undef, vp.nx * vp.ny)
    gp.pcnt = CuArray{Int32,1}(undef, vp.nx * vp.ny)

    gp.a = CuArray{Float64}(undef, (1, 2))

    gp.px = CuArray{Float64}(undef, vp.max_npartcl)#x coordinate of particle
    gp.py = CuArray{Float64}(undef, vp.max_npartcl)#y coordinate of particle
    gp.pT = CuArray{Float64}(undef, vp.max_npartcl)#Temperature of particle
    gp.pPh = CuArray{Int8}(undef, vp.max_npartcl)#???#Ph?

    np_dikes = gp.particle_edges[ndikes_all+1]#number of particles in each dike during intrusion

    gp.px_dikes = CuArray{Float64,1}(undef, np_dikes)#x of dykes particles
    gp.py_dikes = CuArray{Float64,1}(undef, np_dikes)#y of dykes particles

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


    #a and b of ellips for dikes
    gp.dike_a = Array{Float64,1}(undef, ndikes_all)
    gp.dike_b = Array{Float64,1}(undef, ndikes_all)

    #x and y coordinate of center
    gp.dike_x = Array{Float64,1}(undef, ndikes_all)
    gp.dike_y = Array{Float64,1}(undef, ndikes_all)

    #???
    gp.dike_t = Array{Float64,1}(undef, ndikes_all)

    #NOTE:Dykes data upload takes time
    io = open(data_folder*"dikes.bin", "r")
    read!(io, gp.dike_a)
    read!(io, gp.dike_b)
    read!(io, gp.dike_x)
    read!(io, gp.dike_y)
    read!(io, gp.dike_t)

    close(io)

    fid = h5open(data_folder*"particles.h5", "r")

    gp.h_px = Array{Float64,1}(undef, vp.max_npartcl)
    gp.h_py = Array{Float64,1}(undef, vp.max_npartcl)

    gp.h_px = read(fid, "px")
    gp.h_py = read(fid, "py")

    copyto!(gp.px, gp.h_px)
    copyto!(gp.py, gp.h_py)

    #???
    gp.h_px_dikes = Array{Float64,1}(undef, np_dikes)
    gp.h_py_dikes = Array{Float64,1}(undef, np_dikes)

    gp.h_px_dikes = read(fid, "px_dikes")
    gp.h_py_dikes = read(fid, "py_dikes")

    copyto!(gp.px_dikes, gp.h_px_dikes)
    copyto!(gp.py_dikes, gp.h_py_dikes)

    close(fid)

    #=
    	#process markers
    	fid = h5open("data/markers.h5", "r")

    	obj = fid["0"]

    	gp.h_mx = Array{Float64,1}(undef, max_nmarker)
    	gp.h_my = Array{Float64,1}(undef, max_nmarker)
    	gp.h_mT = Array{Float64,1}(undef, max_nmarker)

    	gp.h_mx = read(obj, "mx")
    	gp.h_my = read(obj, "my")
    	gp.h_mT = read(obj, "mT")

    	close(fid)

    	#copyto!(mx, h_mx)
    	#copyto!(my, h_my)
    	#copyto!(mT, h_mT)

    =#


	NDIGITS = Int32(floor(log10(vp.nt)))+1

    filename = data_folder*"grid." * "0"^NDIGITS * ".h5"
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

function init(gp::GridParams, vp::VarParams)
    pic_amount_tmp = vp.pic_amount
    vp.pic_amount = 1.0

    blockSize1D = 768
    gridSize1D = convert(Int64, floor((vp.npartcl + blockSize1D - 1) / blockSize1D))

    #NOTE:
    #changing only pT
    #grid to particles interpolation
    #differene with cuda like 6.e-8 for some reason
    @cuda blocks = gridSize1D threads = blockSize1D g2p!(gp.T, gp.T_old, gp.px, gp.py, gp.pT, vp.dx, vp.dy, vp.pic_amount, vp.nx, vp.ny, vp.npartcl)

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

    #processing all markers 
    gridSize1D = Int64(floor((vp.max_nmarker - vp.nmarker + blockSize1D - 1) / blockSize1D))
    mTs = @view gp.mT[vp.nmarker+1:end]
    @cuda blocks = gridSize1D threads = blockSize1D init_particles_T(mTs, vp.T_magma, vp.max_nmarker - vp.nmarker)

    synchronize()

    vp.pic_amount = pic_amount_tmp

end


function check_melt_fracton(gp::GridParams, vp::VarParams)
    @time begin
        nxl = vp.nxl
        nyl = vp.nyl

        blockSizel = (16, 32)
        gridSizel = (
            (vp.nxl + blockSizel[1] - 1) ÷ blockSizel[1],
            (vp.nyl + blockSizel[2] - 1) ÷ blockSizel[2],
        )

        #Усредняется mf 
        @cuda blocks = gridSizel threads = blockSizel average!(gp.mfl, gp.T, gp.C, vp.nl, vp.nx, vp.ny)

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

        #searching for max vol
        for (idx, vol) in volumes
            if vol > maxVol
                maxVol = vol
                maxIdx = idx
            end
        end

    end

    return maxVol, maxIdx
end


function eruption_advection(gp::GridParams, vp::VarParams, maxVol, maxIdx, iSample, is_eruption, it)
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

        copyto!(cell_idx, cell_idx_host)

        local blockSize1D = 512
        local gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

        #advect particles
        @cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(gp.px, gp.py, cell_idx, vp.gamma, dxl, dyl, vp.npartcl, maxVol, vp.nxl, vp.nyl)
        synchronize()



        gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D
        #advect markers
        @cuda blocks = gridSize1D threads = blockSize1D advect_particles_eruption(gp.mx, gp.my, cell_idx, vp.gamma, dxl, dyl, vp.nmarker, maxVol, vp.nxl, vp.nyl)
        synchronize()

        iSample = iSample + 1

        is_eruption = true
        append!(gp.eruptionSteps, it)
    end

end

function inserting_dykes(gp::GridParams, vp::VarParams, it)
    @time begin
        for i = 1:gp.ndikes[it]
            vp.idike = vp.idike + 1
            idike = vp.idike

            blockSize1D = 512
            gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

            @cuda blocks = gridSize1D threads = blockSize1D advect_particles_intrusion(
                gp.px,
                gp.py,
                gp.dike_a[idike],
                gp.dike_b[idike],
                gp.dike_x[idike],
                gp.dike_y[idike],
                gp.dike_t[idike],
                vp.nu,
                vp.G,
                gp.ndikes[it],
                vp.npartcl
            )

            dike_start = gp.particle_edges[idike]
            dike_end = gp.particle_edges[idike+1]
            np_dike = dike_end - dike_start

            if (vp.npartcl + np_dike > vp.max_npartcl)
                @printf("ERROR: number of particles exceeds maximum value, increase capacity\n")
                return -1
            end


            pxs = @view gp.px[(vp.npartcl+1):(vp.npartcl+np_dike)]
            px_dikess = @view gp.px_dikes[(dike_start+1):(dike_start+np_dike)]
            pys = @view gp.py[(vp.npartcl+1):(vp.npartcl+np_dike)]
            py_dikess = @view gp.py_dikes[(dike_start+1):(dike_start+np_dike)]


            copyto!(pxs, px_dikess)
            copyto!(pys, py_dikess)


            vp.npartcl += np_dike

            gridSize1D = (vp.nmarker + blockSize1D - 1) ÷ blockSize1D


            @cuda blocks = gridSize1D threads = blockSize1D advect_particles_intrusion(
                gp.mx,
                gp.my,
                gp.dike_a[idike],
                gp.dike_b[idike],
                gp.dike_x[idike],
                gp.dike_y[idike],
                gp.dike_t[idike],
                vp.nu,
                vp.G,
                gp.ndikes[it],
                vp.nmarker,
            )

            synchronize()

            vp.nmarker += gp.marker_edges[idike+1] - gp.marker_edges[idike]

        end
    end
end

function p2g_interpolation(gp::GridParams, vp::VarParams)
    @time begin
        fill!(gp.T, 0)
        fill!(gp.C, 0)
        fill!(gp.wts, 0)

        blockSize1D = 512
        gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

        blockSize = (16, 32)
        gridSize = (Int64(floor((vp.nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((vp.ny + blockSize[2] - 1) / blockSize[2])))

        @cuda blocks = gridSize1D threads = blockSize1D p2g_project!(gp.T, gp.C, gp.wts, gp.px, gp.py, gp.pT, gp.pPh, vp.dx, vp.dy, vp.nx, vp.ny, vp.npartcl, vp.npartcl0)
        synchronize()


        @cuda blocks = gridSize threads = blockSize p2g_weight!(gp.T, gp.C, gp.wts, vp.nx, vp.ny)
        synchronize()
    end
end

function particles_injection(gp::GridParams, vp::VarParams)
    @time begin
        blockSize1D = 512
        gridSize1D = (vp.npartcl + blockSize1D - 1) ÷ blockSize1D

        blockSize = (16, 32)
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
