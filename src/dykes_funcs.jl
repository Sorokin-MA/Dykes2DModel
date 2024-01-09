#Function that write which device we use
macro H5Z_FILTER_BLOSC()
	return 5
end

macro GPU_ID()
	return 0
end

macro indices()
	return :(blockIdx().x* blockDim().x + threadIdx().x, blockIdx().y* blockDim().y + threadIdx().y)
end

macro idc(ix, iy)
	return :(iy * nx + ix)
end

#Functon to write number of CUDA device
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


function blerp(x1, x2, y1, y2, f11, f12, f21, f22, x, y)
	invDxDy = 1.0 / ((x2 - x1) * (y2 - y1))

	dx1 = x - x1
	dx2 = x2 - x

	dy1 = y - y1
	dy2 = y2 - y

	return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1)
end

function dmf_basalt(T)
	t1 = T * T
	t11 = exp(0.143636887899999948e3 - 0.2214446257e-6 * t1 * T + t1 * 0.572468110399999928e-3 + T * (-0.494427718499999891e0))
	t14 = (0.1e1 + t11) ^ 0.2e1
	return 0.6643338771e-6 * (t1 - 0.1723434948e4 * T + 0.7442458310e6) * t11 / t14
end

function dmf_rhyolite(T)
	t1 = T * T
	t9 = exp(0.961026e3 - 0.186618e-5 * t1 * T + t1 * 0.447948e-2 + T * (-0.359050e1))
	t12 = (0.1e1 + t9) * (0.1e1 + t9)
	return 0.559856e-5 / t12 * t9 * (t1 - 0.160022e4 * T + 0.641326e6)
end

#mb not working the way it should
function mf_basalt(T)
	T = T / 1000
	t2 = T * T
	t7 = exp(143.636887935970 - 494.427718497039 * T + 572.468110446565 * t2 - 221.444625682461 * t2 * T)
	return 0.1 / (0.1 + t7)
end

#TODO: fix
function mf_rhyolite(T)
	T = T / 1000
	t2 = T * T
	t7 = exp(143.636887935970 - 494.427718497039 * T + 572.468110446565 * t2 - 221.444625682461 * t2 * T)
	return 0.1 / (0.1 + t7)
end

macro mf_magma(T)
	return dmf_rhyolite(T)
end

macro mf_rock(T)
	return mf_rhyolite(T)
end

function update_T(ALL_PARAMS)
	#ix, iy = @indices()
	#TODO:fix indexes
	if ix > nx - 1 || iy > ny - 1
		return
	end

	qxw, qxe, qys, qyn = 0.0, 0.0, 0.0, 0.0

	if ix == 0
		qxw = 0.0
	else
		qxw = -(T[iy * nx + ix] - T[idc(ix - 1, iy)]) / dx
	end

	if ix == nx - 1
		qxe = 0.0
	else
		qxe = -(T[iy * nx + ix +1] - T[iy * nx + ix]) / dx
	end

	if iy == 0
		qys = -2.0 * (T[iy * nx + ix] - T_bot) / dy
	else
		qys = -(T[iy * nx + ix] - T[idc(ix, iy - 1)]) / dy
	end

	if iy == ny - 1
		qyn = -2.0 * (T_top - T[iy * nx + ix]) / dy
	else
		qyn = -(T[((iy + 1) * nx + ix)] - T[iy * nx + ix]) / dy
	end

	dmf = dmf_magma(T[iy * nx + ix]) * C[iy * nx + ix] + dmf_rock(T[iy * nx + ix]) * (1.0 - C[iy * nx + ix])
	lam_rhoCp = (lam_m_rhoCp * C[iy * nx + ix]) + lam_r_rhoCp * (1.0 - C[iy * nx + ix])

	chi = lam_rhoCp / (1.0 + L_Cp * dmf)

	T[iy * nx + ix] += -dt * chi * ((qxe - qxw) / dx + (qyn - qys) / dy)
end

function init_particles_Ph(pPh::Ptr{Int8}, ph::Int8, npartcl::Int)
	ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	if ip > npartcl - 1
		return
	end

	unsafe_store!(pPh, ip, ph)
end

function sign(val)
	return (val > zero(val)) - (val < zero(val))
end

function cart2ellipt(f, x, y)
	xi_eta = zeros(2)
	xi_eta[1] = acosh(max(0.5 / f * (sqrt((x + f) * (x + f) + y * y) + sqrt((x - f) * (x - f) + y * y)), 1.0))
	xi_eta[2] = acos(min(max(x / (f * cosh(xi_eta[1])), -1.0), 1.0)) * sign(y)
	return xi_eta
end

function rot2d(x, y, sb, cb)
	return  (x * cb - y * sb, x * sb + y * cb)
end

function crack_params(a, b, nu, G)

	f = 2 * nu * (a + b) - 2 * a - b

	return (-2 * b * G / f, 0.5 * f / (nu - 1))
end

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
	return u_v
end

function displacements(st, ct, p, s1, s3, f, x, y, nu, G)
	x_y = rot2d(x, y, -st, ct)
	if abs(x_y[1]) < 1e-10
		x_y[1] = 1e-10
	end
	if abs(x_y[2]) < 1e-10
		x_y[2] = 1e-10
	end
	xi_eta = cart2ellipt(f, x_y[1], x_y[2])
	seta = sin(xi_eta[2])
	ceta = cos(xi_eta[2])
	shxi = sinh(xi_eta[1])
	chxi = cosh(xi_eta[1])
	u_v1 = disp_inf_stress(s1 - p, st, ct, f, nu, G, shxi, chxi, seta, ceta)
	u_v2 = disp_inf_stress(s3 - p, ct, -st, f, nu, G, shxi, chxi, seta, ceta)
	I = shxi * seta
	J = chxi * ceta
	u3 = 0.25 * p * f / G * (J * (3.0 - 4.0 * nu) - J)
	v3 = 0.25 * p * f / G * (I * (3.0 - 4.0 * nu) - I)
	u_v = rot2d(u_v1[1] + u_v2[1] + u3, u_v1[2] + u_v2[2] + v3, st, ct)
	u_v[1] = -u_v[1]
	u_v[2] = -u_v[2]
	return u_v
end

function advect_particles_intrusion(px, py, a, b, x, y, theta, nu, G, ndikes, npartcl)
	ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	if ip > npartcl - 1
		return
	end

	p_a0 = crack_params(a, b, nu, G)
	st = sin(theta)
	ct = cos(theta)
	u_v = displacements(st, ct, p_a0[1], 0, 0, p_a0[2], px[ip] - x, py[ip] - y, nu, G)
	px[ip] += u_v[1]
	py[ip] += u_v[2]
	return nothing
end

function p2g_project(ALL_PARAMS)
	ip = blockIdx().x * blockDim().x + threadIdx().x
	
	if ip > npartcl - 1
		return
	end
	
	pxi = px[ip] / dx
	pyi = py[ip] / dy
	
	if pxi < -1 || pxi > nx || pyi < -1 || pyi > ny
		return
	end
	
	ix1 = min(max(Int(pxi), 0), nx - 2)
	iy1 = min(max(Int(pyi), 0), ny - 2)
	ix2 = ix1 + 1
	iy2 = iy1 + 1
	
	k11 = max(1 - abs(pxi - ix1), 0.0) * max(1 - abs(pyi - iy1), 0.0)
	k12 = max(1 - abs(pxi - ix1), 0.0) * max(1 - abs(pyi - iy2), 0.0)
	k21 = max(1 - abs(pxi - ix2), 0.0) * max(1 - abs(pyi - iy1), 0.0)
	k22 = max(1 - abs(pxi - ix2), 0.0) * max(1 - abs(pyi - iy2), 0.0)
	
	atomicAdd(T[idc(ix1, iy1)], k11 * pT[ip])
	atomicAdd(T[idc(ix1, iy2)], k12 * pT[ip])
	atomicAdd(T[idc(ix2, iy1)], k21 * pT[ip])
	atomicAdd(T[idc(ix2, iy2)], k22 * pT[ip])
	
	pC = (pPh == nothing) ? (ip > npartcl0 ? 1.0 : 0.0) : Float64(pPh[ip])
	
	atomicAdd(C[idc(ix1, iy1)], k11 * pC)
	atomicAdd(C[idc(ix1, iy2)], k12 * pC)
	atomicAdd(C[idc(ix2, iy1)], k21 * pC)
	atomicAdd(C[idc(ix2, iy2)], k22 * pC)
	
	atomicAdd(wts[idc(ix1, iy1)], k11)
	atomicAdd(wts[idc(ix1, iy2)], k12)
	atomicAdd(wts[idc(ix2, iy1)], k21)
	atomicAdd(wts[idc(ix2, iy2)], k22)
end

function p2g_weight(ALL_PARAMS)
	ix, iy = @indices()
	
	if ix > nx - 1 || iy > ny - 1
		return
	end
	
	if wts[iy * nx + ix] == 0.0
		return
	end
end

function p2g_project(ALL_PARAMS)
	ip = blockIdx().x * blockDim().x + threadIdx().x
	
	if ip > npartcl - 1
		return
	end
	
	pxi = px[ip] / dx
	pyi = py[ip] / dy
	
	if pxi < -1 || pxi > nx || pyi < -1 || pyi > ny
		return
	end
	
	ix1 = min(max(Int(pxi), 0), nx - 2)
	iy1 = min(max(Int(pyi), 0), ny - 2)
	ix2 = ix1 + 1
	iy2 = iy1 + 1
	
	k11 = max(1 - abs(pxi - ix1), 0.0) * max(1 - abs(pyi - iy1), 0.0)
	k12 = max(1 - abs(pxi - ix1), 0.0) * max(1 - abs(pyi - iy2), 0.0)
	k21 = max(1 - abs(pxi - ix2), 0.0) * max(1 - abs(pyi - iy1), 0.0)
	k22 = max(1 - abs(pxi - ix2), 0.0) * max(1 - abs(pyi - iy2), 0.0)
	
	atomicAdd(T[idc(ix1, iy1)], k11 * pT[ip])
	atomicAdd(T[idc(ix1, iy2)], k12 * pT[ip])
	atomicAdd(T[idc(ix2, iy1)], k21 * pT[ip])
	atomicAdd(T[idc(ix2, iy2)], k22 * pT[ip])
	
	pC = (pPh == nothing) ? (ip > npartcl0 ? 1.0 : 0.0) : Float64(pPh[ip])
	
	atomicAdd(C[idc(ix1, iy1)], k11 * pC)
	atomicAdd(C[idc(ix1, iy2)], k12 * pC)
	atomicAdd(C[idc(ix2, iy1)], k21 * pC)
	atomicAdd(C[idc(ix2, iy2)], k22 * pC)
	
	atomicAdd(wts[idc(ix1, iy1)], k11)
	atomicAdd(wts[idc(ix1, iy2)], k12)
	atomicAdd(wts[idc(ix2, iy1)], k21)
	atomicAdd(wts[idc(ix2, iy2)], k22)
end

function p2g_weight(ALL_PARAMS)
	ix, iy = @indices()
	
	if ix > nx - 1 || iy > ny - 1
		return
	end
	
	if wts[iy * nx + ix] == 0.0
		return
	end
	
	T[iy * nx + ix] /= wts[iy * nx + ix]
	C[iy * nx + ix] /= wts[iy * nx + ix]
end

function blerp(x1, x2, y1, y2, f11, f12, f21, f22, x, y)
	invDxDy = 1.0 / ((x2 - x1) * (y2 - y1))
	
	dx1 = x - x1
	dx2 = x2 - x
	
	dy1 = y - y1
	dy2 = y2 - y
	
	return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1)
end

function g2p(ALL_PARAMS)
	ip = blockIdx.x * blockDim.x + threadIdx.x
	
	if ip > npartcl - 1
		return
	end
	
	pxi = px[ip] / dx
	pyi = py[ip] / dy
	
	ix1 = min(max(Int(pxi), 0), nx - 2)
	iy1 = min(max(Int(pyi), 0), ny - 2)
	ix2 = ix1 + 1
	iy2 = iy1 + 1
	
	x1 = float(ix1) * dx
	x2 = float(ix2) * dx
	y1 = float(iy1) * dy
	y2 = float(iy2) * dy
	
	T_pic = blerp(x1, x2, y1, y2, T[idc(ix1, iy1)], T[idc(ix1, iy2)], T[idc(ix2, iy1)], T[idc(ix2, iy2)], px[ip], py[ip])
	T_flip = pT[ip] + T_pic - blerp(x1, x2, y1, y2, T_old[idc(ix1, iy1)], T_old[idc(ix1, iy2)], T_old[idc(ix2, iy1)], T_old[idc(ix2, iy2)], px[ip], py[ip])
	pT[ip] = T_pic * pic_amount + T_flip * (1.0 - pic_amount)
end

function assignUniqueLables(mf, L, tsh, nx, ny)
	ix, iy = @indices()
	
	if ix > nx - 1 || iy > ny - 1
		return
	end
	
	if mf[ix, iy] >= tsh
		L[iy * nx + ix] = iy * nx + ix
	else
		L[iy * nx + ix] = -1
	end
	return nothing
end

function cwLabel(L, nx, ny)
	iy = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	
	if iy > ny - 1
		return
	end
	
	for ix = nx - 2:-1:0
		if L[iy * nx + ix] >= 0 && L[iy * nx + ix + 1] >= 0
			L[iy * nx + ix] = L[iy * nx + ix + 1]
		end
	end
end

function find_root(L, idx)
	label = idx
	while L[label] != label
		label = L[label]
	end
	return label
end

function merge_labels(L, div, nx, ny)
	iy = (blockIdx().x) * blockDim().x + threadIdx().x
	iy = div ÷ 2 + iy * div - 1
	if iy > ny - 2
		return
	end

	for ix in 1:nx
		# if L[iy * nx + ix] >= 0 && L[((iy + 1) * nx + ix)] >= 0
		#	 lroot = find_root(L, iy * nx + ix)
		#	 rroot = find_root(L, ((iy + 1) * nx + ix))
		#	 L[min(lroot, rroot)] = L[max(lroot, rroot)]
		# end
	end
	return nothing
end

function relabel(L, nx, ny)
	ix, iy = @indices()

	if ix > nx - 1 || iy > ny - 1
		return
	end

	if L[iy * nx + ix] >= 0
		L[iy * nx + ix] = find_root(L, iy * nx + ix)
	end
	return nothing
end


function advect_particles_eruption(px, py, idx, gamma, dxl, dyl, npartcl, ncells, nxl, nyl)
	ip = (blockIdx().x - 1 ) * blockDim().x + threadIdx().x

	if ip > npartcl - 1
		return
	end

	u = 0.0
	v = 0.0

	for i = 1:ncells
		ic = idx[i]
		icx = ic % nxl
		icy = ic ÷ nxl

		xl = icx * dxl
		yl = icy * dyl

		dxl2 = dxl * dxl
		dyl2 = dyl * dyl

		delx = px[ip] - xl
		dely = py[ip] - yl
		r = max(sqrt(delx * delx + dely * dely), sqrt(dxl2 + dyl2))
		r2_2pi = r * r * 2 * π

		u -= dxl2 * (1.0 - gamma) * delx / r2_2pi
		v -= dyl2 * (1.0 - gamma) * dely / r2_2pi
	end

	px[ip] += u
	py[ip] += v
	return nothing
end

function average(mfl, T, C, nl, nx, ny)
	ixl = (blockIdx().x) * blockDim().x + threadIdx().x
	iyl = (blockIdx().y) * blockDim().y + threadIdx().y

	if ixl > (nx ÷ nl - 1) || iyl > (ny ÷ nl - 1)
		return
	end

	avg = 0.0
	for ix = (ixl * nl + 1):(ixl + 1) * nl
		if ix > nx - 1
			break
		end
		for iy = (iyl * nl + 1):(iyl + 1) * nl
			if iy > ny - 1
				break
			end
			vf = C[iy * nx + ix]
			avg = avg + (mf_rhyolite(T[(iy * nx + ix)])) * vf + mf_rhyolite(T[(iy * nx + ix)]) * (1 - vf)
		end
	end
	avg = div(avg, (nl * nl))
	mfl[iyl * (nx ÷ nl) + ixl] = avg
	return nothing;
end

function count_particles(pcnt, px, py, dx, dy, nx, ny, npartcl)
	ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	if ip > npartcl - 1
		return
	end

	pxi = px[ip] / dx
	pyi = py[ip] / dy

	ix = min(max(Int(pxi), 0), nx - 2)
	iy = min(max(Int(pyi), 0), ny - 2)

	atomicAdd(pcnt[iy * nx + ix], 1)
end

function inject_particles(px, py, pT, pPh, npartcl, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl)
	if ix > nx - 2 || iy > ny - 2
		return
	end

	if pcnt[iy * nx + ix] < min_pcount
		for ioy = 1:2
			for iox = 1:2
				inx = ix + iox
				iny = iy + ioy
				new_npartcl = atomicAdd(npartcl, 1)
				if new_npartcl > max_npartcl - 1
					break
				end
				ip = new_npartcl
				px[ip] = dx * inx
				py[ip] = dy * iny

				pT[ip] = T[idc(inx, iny)]
				pPh[ip] = C[idc(inx, iny)] < 0.5 ? 0 : 1
			end
		end
	end
end

function ccl(mf, L, tsh, nx, ny)
	blockSize2D = (16, 32)
	gridSize2D = ((nx + blockSize2D[1] - 1) ÷ blockSize2D[1], (ny + blockSize2D[2] - 1) ÷ blockSize2D[2])
	
	#to synch device
	CUDA.@sync begin
		@cuda blocks = gridSize2D threads = blockSize2D assignUniqueLables(mf, L, tsh, nx, ny)
	end

	blockSize1D = 32
	gridSize1D = (ny + blockSize1D - 1) ÷ blockSize1D
	CUDA.@sync begin
		@cuda  blocks = gridSize1D threads = blockSize1D cwLabel(L, nx, ny)
	end

	div = 2
	npw = convert(Int64,ceil(log2(ny)))
	nyw = 1 << npw
	for i = 1:npw
		gridSize1D = max((nyw + blockSize1D - 1) ÷ blockSize1D ÷ div, 1)
		#CUDA.@sync begin
			@cuda blocks = gridSize1D threads = blockSize1D merge_labels(L, div, nx, ny)
		#end
		div *= 2
	end
	
	CUDA.@sync begin
		@cuda blocks = gridSize2D threads = blockSize2D relabel(L, nx, ny)
	end

end

function printDeviceProperties(deviceId)
	# code to print device properties
end

function tic()
	return time()
end

function toc(start)
	end_time = time()
	elapsed_time = end_time - start
	println(elapsed_time, " s")
end

#=
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
