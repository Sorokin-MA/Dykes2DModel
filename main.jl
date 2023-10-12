#Pkg.add("BenchmarkTools")
#Pkg.add("HDF5")
#Pkg.add("CUDA")
#Pkg.add("JupyterFormatter")


using CUDA
using Printf
using BenchmarkTools

function kernel()
	dev = Ref{Cint}()
	CUDA.cudaGetDevice(dev)
	@cuprintln("Running on device $(dev[])")
	return
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

using HDF5
using Printf

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
	ip = blockIdx().x * blockDim().x + threadIdx().x

	if ip > (npartcl - 1)
		return nothing
	end

	pxi = px[ip] / dx
	pyi = px[ip] / dy

	ix1 = min(max(convert(Int64, pxi), 0), nx - 2)
	iy1 = min(max(convert(Int64, pyi), 0), ny - 2)

	ix2 = ix1 + 1
	iy2 = iy1 + 1

	x1 = ix1 * dx
	x2 = ix2 * dx
	y1 = iy1 * dy
	y2 = iy2 * dy

	T_pic = blerp(x1, x2, y1, y2, T[idc(ix1, iy1, nx)], T[idc(ix1, iy2, nx)], T[idc(ix2, iy1, nx)], T[idc(ix2, iy2, nx)], px[ip], py[ip])
	T_flip = pT[ip] + T_pic - blerp(x1, x2, y1, y2, T_old[idc(ix1, iy1, nx)], T_old[idc(ix1, iy2, nx)], T_old[idc(ix2, iy1, nx)], T_old[idc(ix2, iy2, nx)], px[ip], py[ip])
	pT[ip] = T_pic * pic_amount + T_flip * (1.0 - pic_amount)

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
	for ix ∈ (ixl*nl):((ixl+1)*nl)

		if (ix > nx - 1)
			break
		end

		#for (int iy = iyl * nl; iy < (iyl + 1) * nl; ++iy) 
		for iy ∈ (iyl*nl):(iy<(iyl+1)*nl)
			if (iy > ny - 1)
				break
			end
			vf = C[idc(ix, iy, nx)]
			avg += mf_magma(T[idc(ix, iy, nx)]) * vf + mf_rock(T[idc(ix, iy, nx)]) * (1 - vf)
			#=  
			avg += mf_magma(T[idc(ix, iy, nx)]) * vf + mf_rock(T[idc(ix, iy, nx)]) * (1 - vf);
			=#
		end
	end

	avg /= nl * nl
	mfl[iyl*(nx/nl)+ixl] = avg

end

macro CUINDICES()
	\
	ix = blockIdx().x * blockDim().x + threadIdx().x
	\
	iy = blockIdx().y * blockDim().y + threadIdx().y
end


function assignUniqueLables(mf, L, tsh, nx, ny)
	ix = blockIdx().x * blockDim().x + threadIdx().x
	\
	iy = blockIdx().y * blockDim().y + threadIdx().y

	if (ix > nx - 1 || iy > ny - 1)
		return
	end

	if (mf[idc(ix, iy, nx)] >= tsh)
		L[idc(ix, iy, nx)] = idc(ix, iy, nx)
	else
		L[idc(ix, iy, nx)] = -1
	end
	return
end

#=
__global__ void assignUniqueLables(double *mf, int *L, double tsh, int nx, int ny) {
  CUINDICES

  if (ix > nx - 1 || iy > ny - 1) {
	return;
  }

  if (mf[idc(ix, iy)] >= tsh) {
	L[idc(ix, iy)] = idc(ix, iy);
  } else {
	L[idc(ix, iy)] = -1;
  }
}
=#
function cwLabel(L, nx, ny)
	iy = blockIdx().x * blockDim().x + threadIdx().x

	if (iy > ny - 1)
		return
	end

end

#=
cwLabel(int *L, int nx, int ny) {
  int iy = blockIdx.x * blockDim.x + threadIdx.x;

  if (iy > ny - 1) {
	return;
  }
=#

function merge_labels(L, div, nx, ny)
	return
	iy = blockIdx.x * blockDim.x + threadIdx.x

	iy = div / 2 + iy * div - 1
	if (iy > ny - 2)
		return
	end
	#=
		for (int ix = 0; ix < nx; ++ix) {
			if (L[idc(ix, iy)] >= 0 && L[idc(ix, iy + 1)] >= 0) {
			  int lroot = find_root(L, idc(ix, iy));
			  int rroot = find_root(L, idc(ix, iy + 1));
			  L[min(lroot, rroot)] = L[max(lroot, rroot)];
			}
		  }
		  =#

	for ix ∈ 0:nx
		if (L[idc(ix, iy, nx)] >= 0 && L[idc(ix, iy + 1, nx)] >= 0)
			lroot = find_root(L, idc(ix, iy, nx))
			rroot = find_root(L, idc(ix, iy + 1, nx))
			L[min(lroot, rroot)] = L[max(lroot, rroot)]
		end
	end

	return
end

#=
__global__ void merge_labels(int *L, int div, int nx, int ny) {
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  iy = div / 2 + iy * div - 1;
  if (iy > ny - 2) {
	return;
  }
=#
function find_root(L, idx)
	label = idx
	while (L[label] != label)
		label = L[label]
	end
	return label
end

#=
__device__ int find_root(int *L, int idx) {
  int label = idx;
  while (L[label] != label) {
	label = L[label];
  }
  return label;
}
=#

function relabel(L, nx, ny)
	ix = blockIdx().x * blockDim().x + threadIdx().x
	iy = blockIdx().y * blockDim().y + threadIdx().y

	if (ix > nx - 1 || iy > ny - 1)
		return
	end

	if (L[idc(ix, iy, nx)] >= 0)
		L[idc(ix, iy, nx)] = find_root(L, idc(ix, iy, nx))
	end
	return
end

#=
__global__ void relabel(int *L, int nx, int ny) {
  CUINDICES

  if (ix > nx - 1 || iy > ny - 1) {
	return;
  }

  if (L[idc(ix, iy)] >= 0) {
	L[idc(ix, iy)] = find_root(L, idc(ix, iy));
  }
}
=#
function ccl(mf, L, tsh, nx, ny)
	blockSize2D = (16, 32)
	gridSize2D = (convert(Int64, floor((nx + blockSize2D[1] - 1) / blockSize2D[1])), convert(Int64, floor((ny + blockSize2D[2] - 1) / blockSize2D[2])))


	@cuda blocks = gridSize2D threads = blockSize2D assignUniqueLables(mf, L, tsh, nx, ny)
	#assignUniqueLables<<<gridSize2D, blockSize2D>>>(mf, L, tsh, nx, ny);
	synchronize()

	blockSize1D = 32
	gridSize1D = convert(Int64, floor((ny + blockSize1D - 1) / blockSize1D))

	@cuda blocks = gridSize1D threads = blockSize1D cwLabel(L, nx, ny)
	#cwLabel<<<gridSize1D, blockSize1D>>>(L, nx, ny);
	synchronize()

	div = 2
	npw = convert(Int64, ceil(log2(ny)))
	nyw = 1 << npw

	for i ∈ 0:npw
		gridSize1D = (convert(Int64, floor(max((nyw + blockSize1D - 1) / blockSize1D / div, 1))))
		@cuda blocks = gridSize1D threads = blockSize1D merge_labels(L, div, nx, ny)
		synchronize()
		div *= 2
	end
	@cuda blocks = gridSize2D threads = blockSize2D relabel(L, nx, ny)
	#relabel<<<gridSize2D, blockSize2D>>>(L, nx, ny);
	synchronize()

end

function write_h5(fid, name, arr, buff,sizze)
	#@inbounds copyto!(arr, buff)
	#buff[1:sizze] .= arr[1:sizze]
	copyto!(buff[1:sizze], arr[1:sizze])
	write(fid,name,buff[1:sizze])
end

#=
static void ccl(double *mf, int *L, double tsh, int nx, int ny) {
  dim3 blockSize2D(16, 32);
  dim3 gridSize2D((nx + blockSize2D.x - 1) / blockSize2D.x, (ny + blockSize2D.y - 1) / blockSize2D.y);

  assignUniqueLables<<<gridSize2D, blockSize2D>>>(mf, L, tsh, nx, ny);
  CUCHECK(cudaDeviceSynchronize());

  dim3 blockSize1D(32);
  dim3 gridSize1D((ny + blockSize1D.x - 1) / blockSize1D.x);

  cwLabel<<<gridSize1D, blockSize1D>>>(L, nx, ny);
  CUCHECK(cudaDeviceSynchronize());

  int div = 2;
  int npw = int(ceil(log2(ny)));
  int nyw = 1 << npw;
  for (int i = 0; i < npw; ++i) {
	gridSize1D.x = max((nyw + blockSize1D.x - 1) / blockSize1D.x / div, 1);
	merge_labels<<<gridSize1D, blockSize1D>>>(L, div, nx, ny);
	CUCHECK(cudaDeviceSynchronize());
	div *= 2;
  }

  relabel<<<gridSize2D, blockSize2D>>>(L, nx, ny);
  CUCHECK(cudaDeviceSynchronize());
}
=#

#using JupyterFormatter
#enable_autoformat()
#disable_autoformat()

function advect_particles_eruption(px, py, idx, gamma, dxl, dyl, npartcl, ncells, nxl, nyl)
	ip = blockIdx().x * blockDim().x + threadIdx().x;

	if ip > npartcl - 1
		return
	end
	u = 0.0;
	v = 0.0;

	for i = 1:ncells
		ic = idx[i];
		icx = ic % nxl;
		icy = ic ÷ nxl;

		xl = icx * dxl;
		yl = icy * dyl;
	
		dxl2 = dxl * dxl;
		dyl2 = dyl * dyl;
	
		delx = px[ip] - xl;
		dely = py[ip] - yl;

		r = max(sqrt(delx * delx + dely * dely), sqrt(dxl2 + dyl2));
		r2_2pi = r * r * 2 * pi;
	
		u = u - dxl2 * (1.0 - gamma) * delx ÷ r2_2pi;
		v = v - dyl2 * (1.0 - gamma) * dely ÷ r2_2pi;
	end

	px[ip] = px[ip] + u;
	py[ip] = py[ip] + v;
	
	return
end


function crack_params(a, b, nu, G)
	f = 2 * nu * (a + b) - 2 * a - b;
	p_a0_x =  -2 * b * G / f;
	p_a0_y = 0.5 * f / (nu - 1);
	return (p_a0_x, p_a0_y);
end

#=
__forceinline__ __device__ double2 crack_params(double a, double b, double nu, double G) {
  double2 p_a0;
  double f = 2 * nu * (a + b) - 2 * a - b;
  p_a0.x = -2 * b * G / f;
  p_a0.y = 0.5 * f / (nu - 1);
  return p_a0;
}
=#
function rot2d(x, y, sb, cb)
	x_y_1 = x * cb - y * sb;
	x_y_2 = x * sb + y * cb;
	return (x_y_1, x_y_2);
end

#=
__forceinline__ __device__ double2 rot2d(double x, double y, double sb, double cb) {
  double2 x_y;
  x_y.x = x * cb - y * sb;
  x_y.y = x * sb + y * cb;
  return x_y;
}
=#

function cart2ellipt(f, x, y)
	xi_eta_1 = acosh(max(0.5 / f * (sqrt((x + f) * (x + f) + y * y) + sqrt((x - f) * (x - f) + y * y)), 1.0));
	xi_eta_2 = acos(min(max(x / (f * cosh(xi_eta_1)), -1.0), 1.0)) * sign(y);
	return xi_eta_1, xi_eta_2
end

#=
__forceinline__ __device__ double2 cart2ellipt(double f, double x, double y) {
  double2 xi_eta;
  xi_eta.x = acosh(fmax(0.5 / f * (sqrt((x + f) * (x + f) + y * y) + sqrt((x - f) * (x - f) + y * y)), 1.0));
  xi_eta.y = acos(fmin(fmax(x / (f * cosh(xi_eta.x)), -1.0), 1.0)) * sign(y);
  return xi_eta;
}
=#

function disp_inf_stress(s, st, ct, c, nu, G, shxi, chxi, seta, ceta)
	e2xi0 = 1.0;
	s2b = 2.0 * st * ct;
	c2b = 2.0 * ct * ct - 1.0;
	sh2xi0 = 0.0;
	ch2xi0 = 1.0;
	sh2xi = 2.0 * shxi * chxi;
	ch2xi = 2.0 * chxi * chxi - 1.0;
	s2eta = 2.0 * seta * ceta;
	c2eta = 2.0 * ceta * ceta - 1.0;
	K = 3.0 - 4.0 * nu;
	n = ch2xi - c2eta;
	hlda = e2xi0 * c2b * (K * sh2xi - K * ch2xi + K * c2eta + ch2xi - sh2xi + c2eta) + K * (ch2xi - c2eta) - ch2xi - c2eta + 2.0 * ch2xi0 - 2.0 * c2b + 2.0 * e2xi0 * (c2eta * c2b + s2eta * s2b) * (ch2xi0 * sh2xi - sh2xi0 * ch2xi);
	hldb = e2xi0 * (c2b * (K * s2eta - s2eta) + s2b * (K * ch2xi - K * c2eta + ch2xi + c2eta) - 2.0 * (c2eta * s2b - s2eta * c2b) * (ch2xi0 * ch2xi - sh2xi0 * sh2xi));

	u_v_1 = s * c / (8.0 * n * G) * (hlda * shxi * ceta + hldb * chxi * seta);
	u_v_2 = s * c / (8.0 * n * G) * (hlda * chxi * seta - hldb * shxi * ceta);
	return (u_v_1, u_v_2);
end

#=
__forceinline__ __device__ double2 disp_inf_stress(double s, double st, double ct, double c, double nu, double G, double shxi, double chxi, double seta,
                                                   double ceta) {
  double e2xi0 = 1.0;
  double s2b = 2.0 * st * ct;
  double c2b = 2.0 * ct * ct - 1.0;
  double sh2xi0 = 0.0;
  double ch2xi0 = 1.0;
  double sh2xi = 2.0 * shxi * chxi;
  double ch2xi = 2.0 * chxi * chxi - 1.0;
  double s2eta = 2.0 * seta * ceta;
  double c2eta = 2.0 * ceta * ceta - 1.0;
  double K = 3.0 - 4.0 * nu;
  double n = ch2xi - c2eta;
  double hlda = e2xi0 * c2b * (K * sh2xi - K * ch2xi + K * c2eta + ch2xi - sh2xi + c2eta) //
                + K * (ch2xi - c2eta) - ch2xi - c2eta + 2.0 * ch2xi0 - 2.0 * c2b          //
                + 2.0 * e2xi0 * (c2eta * c2b + s2eta * s2b) * (ch2xi0 * sh2xi - sh2xi0 * ch2xi);
  double hldb = e2xi0 * (c2b * (K * s2eta - s2eta) + s2b * (K * ch2xi - K * c2eta + ch2xi + c2eta) //
                         - 2.0 * (c2eta * s2b - s2eta * c2b) * (ch2xi0 * ch2xi - sh2xi0 * sh2xi));
  double2 u_v;
  u_v.x = s * c / (8.0 * n * G) * (hlda * shxi * ceta + hldb * chxi * seta);
  u_v.y = s * c / (8.0 * n * G) * (hlda * chxi * seta - hldb * shxi * ceta);
  return u_v;
}
=#

function displacements(st, ct, p, s1, s3, f, x, y, nu, G)
	x_y_1, x_y_2 = rot2d(x, y, -st, ct);
	if (abs(x_y_1) < 1.e-10)
		x_y_1 = 1.e-10;
	end
	
	if (abs(x_y_2) < 1e-10)
		x_y_2 = 1.e-10;
	end

	u_v_1, u_v_2 = 1,2

	xi_eta_1, xi_eta_2 = cart2ellipt(f, x_y_1, x_y_2);
	seta = sin(xi_eta_2);
	ceta = cos(xi_eta_2);
	shxi = sinh(xi_eta_1);
	chxi = cosh(xi_eta_1);
	u_v1_1, u_v1_2 = disp_inf_stress(s1 - p, st, ct, f, nu, G, shxi, chxi, seta, ceta);
	u_v2_1, u_v2_2 = disp_inf_stress(s3 - p, ct, -st, f, nu, G, shxi, chxi, seta, ceta);
	I = shxi * seta;
	J = chxi * ceta;
	u3 = 0.25 * p * f / G * (J * (3.0 - 4.0 * nu) - J);
	v3 = 0.25 * p * f / G * (I * (3.0 - 4.0 * nu) - I);

	u_v = rot2d(u_v1_1 + u_v2_1 + u3, u_v1_2 + u_v2_2 + v3, st, ct);
	u_v_1 = -u_v_1;
	u_v_2 = -u_v_2;

	return u_v_1, u_v_2
end


#=
__forceinline__ __device__ double2 displacements(double st, double ct, double p, double s1, double s3, double f, double x, double y, double nu, double G) {
  double2 x_y = rot2d(x, y, -st, ct);
  if (fabs(x_y.x) < 1e-10) {
    x_y.x = 1e-10;
  }
  if (fabs(x_y.y) < 1e-10) {
    x_y.y = 1e-10;
  }
  double2 xi_eta = cart2ellipt(f, x_y.x, x_y.y);
  double seta = sin(xi_eta.y);
  double ceta = cos(xi_eta.y);
  double shxi = sinh(xi_eta.x);
  double chxi = cosh(xi_eta.x);
  double2 u_v1 = disp_inf_stress(s1 - p, st, ct, f, nu, G, shxi, chxi, seta, ceta);
  double2 u_v2 = disp_inf_stress(s3 - p, ct, -st, f, nu, G, shxi, chxi, seta, ceta);
  double I = shxi * seta;
  double J = chxi * ceta;
  double u3 = 0.25 * p * f / G * (J * (3.0 - 4.0 * nu) - J);
  double v3 = 0.25 * p * f / G * (I * (3.0 - 4.0 * nu) - I);
  double2 u_v = rot2d(u_v1.x + u_v2.x + u3, u_v1.y + u_v2.y + v3, st, ct);
  u_v.x = -u_v.x;
  u_v.y = -u_v.y;
  return u_v;
}
=#

function advect_particles_intrusion(px, py, a, b, x, y, theta, nu, G, ndikes, npartcl)
	ip = blockIdx().x * blockDim().x + threadIdx().x;

	if (ip > npartcl - 1)
		return;
	end
	p_a0 = crack_params(a, b, nu, G);
	st = sin(theta);
	ct = cos(theta);

	u_v_1, u_v_2 = displacements(st, ct, p_a0[1], 0, 0, p_a0[2], px[ip] - x, py[ip] - y, nu, G);
	#u_v_1, u_v_2 = 1, 1

	px[ip] = px[ip] + u_v_1;
 	py[ip] = py[ip] + u_v_2;
	return
end
#=
__global__ void advect_particles_intrusion(double *px, double *py, double a, double b, double x, double y, double theta, double nu, double G, int ndikes,
                                           int npartcl) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  double2 p_a0 = crack_params(a, b, nu, G);
  double st = sin(theta);
  double ct = cos(theta);
  double2 u_v = displacements(st, ct, p_a0.x, 0, 0, p_a0.y, px[ip] - x, py[ip] - y, nu, G);
  px[ip] += u_v.x;
  py[ip] += u_v.y;
}
=#


function count_particles(pcnt, px, py, dx, dy, nx, ny, npartcl)
	ip = blockIdx().x * blockDim().x + threadIdx().x;

	if (ip > npartcl - 1)
	  return;
	end

	pxi = px[ip] ÷ dx;
	pyi = py[ip] ÷ dy;

	ix = min(max(pxi, 0), nx - 2);
	iy = min(max(pyi, 0), ny - 2);

	#pcnt[idc(ix, iy, nx)] = pcnt[idc(ix,iy, nx)] + 1;
	#atomicAdd(pcnt[idc(Int32(ix), Int32(iy), nx)], 1);
	#atomic_add!(pcnt[idc(Int32(ix), Int32(iy), nx)], 1);
	#pcnt[idc(ix, iy, nx)] = CUDA.atomic_add!(1, 1);
	#@atomic pcnt[idc(Int32(ix), Int32(iy), nx)] = pcnt[idc(Int32(ix), Int32(iy), nx)] + 1
	return
end

#=
__global__ void count_particles(int *pcnt, double *px, double *py, double dx, double dy, int nx, int ny, int npartcl) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  double pxi = px[ip] / dx;
  double pyi = py[ip] / dy;

  int ix = min(max(int(pxi), 0), nx - 2);
  int iy = min(max(int(pyi), 0), ny - 2);

  atomicAdd(&pcnt[idc(ix, iy)], 1);
}
=#


using Random: Random


function main()


	Random.seed!(1234)

	#Настроить фильтр

	#Настроить девайс если не выбран

	#Вывести свойства
	print_gpu_properties()

	dpa = Array{Float64, 1}(undef, 18)
	ipa = Array{Int32, 1}(undef, 12)

	Lx = 0.0
	Ly = 0.0
	lam_r_rhoCp = 0.0
	lam_m_rhoCp = 0.0
	L_Cp = 0.0
	T_top = 0.0
	T_bot = 0.0
	T_magma = 0.0
	tsh = 0.0
	gamma = 0.0
	Ly_eruption = 0.0
	nu = 0.0
	G = 0.0
	dt = 0.0
	dx = 0.0
	dy = 0.0
	eiter = 0.0
	pic_amount = 0.0

	pmlt = 0
	nx = 0
	ny = 0
	nl = 0
	nt = 0
	niter = 0
	nout = 0
	nsub = 0
	nerupt = 0
	npartcl = 0
	nmarker = 0
	nSample = 0

	#char filename[1024];
	filename = Array{Char, 1}(undef, 1024)

	io = open("pa.bin", "r")
	read!(io, dpa)
	read!(io, ipa)


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

	critVol = Array{Float64, 1}(undef, nSample)
	read!(io, critVol)

	ndikes = Array{Int32, 1}(undef, nt)
	read!(io, ndikes)

	ndikes_all = 0

	for istep ∈ 1:nt
		ndikes_all = ndikes_all + ndikes[istep]
	end

	particle_edges = Array{Int32, 1}(undef, ndikes_all + 1)
	read!(io, particle_edges)

	marker_edges = Array{Int32, 1}(undef, ndikes_all + 1)
	read!(io, marker_edges)

	close(io)

	cap_frac = 1.5
	npartcl0 = npartcl
	max_npartcl = convert(Int64, npartcl * cap_frac) + particle_edges[ndikes_all]

	println(max_npartcl)

	nmarker0 = nmarker

	max_nmarker = nmarker + marker_edges[ndikes_all]


	#dim3 blockSize(16, 32);
	# dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

	T = CuArray{Float64, 2}(undef, nx, ny)
	T_old = CuArray{Float64, 2}(undef, nx, ny)
	C = CuArray{Float64, 2}(undef, nx, ny)
	wts = CuArray{Float64, 2}(undef, nx, ny)
	pcnt = CuArray{Int32, 2}(undef, nx, ny)

	a = CuArray{Float64}(undef, (1, 2))

	px = CuArray{Float64}(undef, max_npartcl)
	py = CuArray{Float64}(undef, max_npartcl)
	pT = CuArray{Float64}(undef, max_npartcl)
	pPh = CuArray{Int8}(undef, max_npartcl)

	np_dikes = particle_edges[ndikes_all]

	px_dikes = CuArray{Float64, 1}(undef, np_dikes)
	py_dikes = CuArray{Float64, 1}(undef, np_dikes)

	mx = CuArray{Float64, 1}(undef, max_nmarker)
	my = CuArray{Float64, 1}(undef, max_nmarker)
	mT = CuArray{Float64, 1}(undef, max_nmarker)

	staging = Array{Float64, 1}(undef, max_npartcl)
	npartcl_d = Array{Int32, 1}(undef, 1)


	nxl = convert(Int64, nx / nl)
	nyl = convert(Int64, ny / nl)

	println("nx  - $nx");
	println("ny - $ny");
	println("nl - $nl");
	println(nxl);
	L = CuArray{Int32, 1}(undef, nxl*nyl)

	L_host = Array{Int32, 1}(undef, nxl*nyl)

	mfl = CuArray{Float64, 2}(undef, nxl, nyl)

	dike_a = Array{Float64, 1}(undef, ndikes_all)
	dike_b = Array{Float64, 1}(undef, ndikes_all)
	dike_x = Array{Float64, 1}(undef, ndikes_all)
	dike_y = Array{Float64, 1}(undef, ndikes_all)
	dike_t = Array{Float64, 1}(undef, ndikes_all)

	#=
	io = open("dikes.bin", "r");
	read!(io, dike_a)
	read!(io, dike_b)
	read!(io, dike_x)
	read!(io, dike_y)
	read!(io, dike_t)

	close(io)

	fid = h5open("particles.h5", "r")
	px = read(fid,"px")
	py = read(fid,"py")
	px_dikes = read(fid,"px_dikes")
	py_dikes = read(fid,"py_dikes")
	close(fid)

	fid = h5open("markers.h5", "r")

	obj = fid["0"]
	read(obj, "mx")
	read(obj, "my")
	read(obj, "mT")

	close(fid)
	=#

	NDIGITS = 4

	filename = "grid." * "0"^NDIGITS * "0" * ".h5"

	fid = h5open(filename, "r")
	T_h = read(fid, "T")
	copyto!(T_h, T)
	C_h = read(fid, "C")
	copyto!(C_h, C)
	close(fid)

	#auto tm_all = tic();

	bar1 = "├──"
	bar2 = "\t ├──"
	#bar2 = "\xb3  \xc3\xc4\xc4";

	@time begin
		#init
		@time begin
			@printf("%s initialization              ", bar1)
			pic_amount_tmp = pic_amount
			pic_amount = 1.0

			blockSize1D = 768
			gridSize1D = convert(Int64, floor((npartcl + blockSize1D - 1) / blockSize1D))


			#gridSize1D

			#@device_code_warntype


			#@cuda blocks = gridSize1D threads=blockSize1D g2p(@ALL_ARGS())
			#kekw = idc(2, 4, nx)
			#println("$kekw");

			#@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)

			gridSize1D = convert(
				Int64,
				floor((max_npartcl - npartcl + blockSize1D - 1) / blockSize1D),
			)

			#@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(pT, T_magma, max_npartcl, npartcl);
			#@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(pPh, 1, max_npartcl, npartcl);  
			gridSize1D = (max_nmarker - nmarker + blockSize1D - 1) / blockSize1D
			#@cuda blocks = gridSize1D threads=blockSize1D init_particles_T(mT, T_magma, max_nmarker, nmarker); 
			synchronize()

			pic_amount = pic_amount_tmp
		end

		idike = 1
		iSample = Int32(1)

		eruptionSteps = Vector{Int32}()

		for it ∈ 1:nt
			#action
			@printf("%s it = %d", bar1, it)
			is_eruption = false
			is_intrusion = (ndikes[it] > 0)

			if (it % nerupt == 0)
				@time begin
					@printf("\n%s checking melt fraction   | ", bar2)

					blockSizel = (16, 32)
					gridSizel = ((nxl + blockSizel[1] - 1) / blockSizel[1], (nyl + blockSizel[2] - 1) / blockSizel[2])

					#не работает
					#@cuda blocks = gridSizel threads=blockSizel average(mfl, T, C, nl, nx, ny);

					#average<<<gridSizel, blockSizel>>>(mfl, T, C, nl, nx, ny);
					synchronize()

					ccl(mfl, L, tsh, nxl, nyl)

					copyto!(L, L_host)
					#cudaMemcpy(L_host, L, SIZE_2D(nxl, nyl, int), cudaMemcpyDeviceToHost)

					volumes = Dict{Int32, Int32}(-1 => 0)

					#println("\n");
					#println(volumes);

					for  iy = 0:(nyl-1)
						if (iy * dy * nl < Ly_eruption) 
							continue;
						end
						for ix = 1:nxl
							if L_host[iy * nxl + ix] >= 0
								if haskey(volumes, L_host[iy * nxl + ix])
								volumes[L_host[iy * nxl + ix]] = volumes[L_host[iy * nxl + ix]] + 1;
								else
								volumes[L_host[iy * nxl + ix]] = 0
								volumes[L_host[iy * nxl + ix]] = volumes[L_host[iy * nxl + ix]] + 1;
								end
							end
						end
					end
		
					maxVol = -1;
					maxIdx = -1;

					for (idx, vol) in volumes
						if vol > maxVol
							maxVol = vol;
							maxIdx = idx;
						end
					end

				end

				dxl = dx * nl
				dyl = dy * nl

				#if (maxVol * dxl * dyl >= critVol[iSample])
					if (true)
					@printf("%s erupting %07d cells   | ", bar2, maxVol)
					@time begin

						cell_idx = CuArray{Int32, 1}(undef, maxVol)
						cell_idx_host = Array{Int32, 1}(undef, maxVol)
						next_idx = 0;



						for  idx = 1: nxl*nyl
							if L_host[idx] == maxIdx
								if next_idx < maxVol
									next_idx = next_idx +1;
									cell_idx_host[next_idx] = idx;
								end
							end
						end

						copyto!(cell_idx, cell_idx_host)

						blockSize1D = 512;
						gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D;
						#advect_particles_eruption<<<gridSize1D, blockSize1D>>>(px, py, cell_idx, gamma, dxl, dyl, npartcl, maxVol, nxl, nyl);
						@cuda blocks = gridSize1D threads=blockSize1D advect_particles_eruption(px, py, cell_idx, gamma, dxl, dyl, npartcl, 5, nxl, nyl);
						synchronize();

						gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D;

						#advect_particles_eruption<<<gridSize1D, blockSize1D>>>(mx, my, cell_idx, gamma, dxl, dyl, nmarker, maxVol, nxl, nyl);
						@cuda blocks = gridSize1D threads=blockSize1D advect_particles_eruption(mx, my, cell_idx, gamma, dxl, dyl, nmarker, 5, nxl, nyl);
						synchronize();

						iSample = iSample + 1;

						is_eruption = true;
						append!( eruptionSteps, it)

					end
				end
			end

			if (is_intrusion)
				@printf("%s inserting %02d dikes       | ", bar2, ndikes[it])
				@time begin
					for i = 1:ndikes[it]
					idike = idike + 1
					blockSize1D = 512;
					gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D;
					#advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(px, py, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
					#									ndikes[it - 1], npartcl);
					@cuda blocks = gridSize1D threads=blockSize1D advect_particles_intrusion(px, py, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G, ndikes[it], npartcl);

					dike_start = particle_edges[idike];
					dike_end = particle_edges[idike + 1];
					np_dike = dike_end - dike_start;
					
					if (npartcl + np_dike > max_npartcl)
						fprintf(stderr, "ERROR: number of particles exceeds maximum value, increase capacity\n");
						exit(EXIT_FAILURE);
					end
					

					#copyto!(px[npartcl], px_dikes[dike_start])
					#px_dikes[dike_start] = px[npartcl]
					#py_dikes[dike_start] = py[npartcl]
					#copyto!(py[npartcl], py_dikes[dike_start])
					npartcl += np_dike;

					gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D;


					#advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(mx, my, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
					#ndikes[it - 1], nmarker);
					@cuda blocks = gridSize1D threads=blockSize1D advect_particles_intrusion(mx, my, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G, ndikes[it], nmarker);
					synchronize();
					nmarker += marker_edges[idike + 1] - marker_edges[idike];
					end
				end
			end



			if (is_eruption || is_intrusion)
				@printf("%s p2g interpolation        | ", bar2)
				@time begin

					#fill!(T, nx*ny)
					#T = fill(Float64, nx*ny)
					T = CUDA.zeros(Float64, nx*ny)
					C = CUDA.zeros(Float64, nx*ny)
					wts = CUDA.zeros(Float64, nx*ny)
		
					blockSize1D = 768
					gridSize1D = (npartcl + blockSize1D - 1) / blockSize1D;

					#p2g_project<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
					synchronize();
					#p2g_weight<<<gridSize, blockSize>>>(ALL_ARGS);
					synchronize();

				end

				@printf("%s particle injection       | ", bar2)
				@time begin


					blockSize1D = 512;
					gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D;
					pcnt = CUDA.zeros(Float64, nx*ny)
					#count_particles<<<gridSize1D, blockSize1D>>>(pcnt, px, py, dx, dy, nx, ny, npartcl);
					#@cuda blocks = gridSize1D threads=blockSize1D count_particles(pcnt, px, py, dx, dy, nx, ny, npartcl);
					synchronize();

					npartcl_d[1] = npartcl
					min_pcount = 2;
					#inject_particles<<<gridSize, blockSize>>>(px, py, pT, pPh, npartcl_d, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl);
					synchronize();
					new_npartcl = npartcl;

					npartcl = npartcl_d[1]

					if new_npartcl > max_npartcl
						fprintf(stderr, "ERROR: number of particles exceeds maximum value, increase capacity\n");
						exit(EXIT_FAILURE);
					end
					if (new_npartcl > npartcl) 
						@printf("(%03d) | ", new_npartcl - npartcl);
						npartcl = new_npartcl;
					else 
						@printf("(000) | ");
					end

				end
			end




			@time begin
				@printf("%s solving heat diffusion   | ", bar2)

				copyto!(T, T_old)
				for isub = 0:nsub
					#update_T<<<gridSize, blockSize>>>(ALL_ARGS);
					synchronize();
				end
			end

			@time begin
				@printf("%s g2p interpolation        | ", bar2)
				#particles g2p
				blockSize1D = 1024;
				gridSize1D = (npartcl + blockSize1D - 1) ÷ blockSize1D;
				#g2p<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
				#@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)

				#markers g2p
				gridSize1D = (nmarker + blockSize1D - 1) ÷ blockSize1D;
				pic_amount_tmp = pic_amount;
				pic_amount = 1.0;
				#g2p<<<gridSize1D, blockSize1D>>>(T, T_old, C, wts, mx, my, mT, NULL, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, nmarker,
				#			 nmarker0);
				#@cuda blocks = gridSize1D threads=blockSize1D g2p!(T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0)
				synchronize();
				pic_amount = pic_amount_tmp;

			end

			if (it % nout == 0 || is_eruption)
				#auto tm = tic();
				@time begin
					@printf("%s writing results to disk  | ", bar2)
					filename = "grid." * string(it) * ".h5"
					fid = h5open(filename, "w")

					write_h5(fid, "T", T, staging, nx * ny)
					write_h5(fid, "C", C, staging, nx * ny)
					
					if (is_eruption) 
						write_h5(fid, "L", L, staging, nxl * nyl)
					end

					close(fid);
				end
			end
;

			@time begin
			@printf("%s writing markers to disk  | ", bar2);
			filename = "markers.h5"
			fid = h5open(filename, "w")
			filename_gname = string(it) * "_gname"
			create_group(fid, filename_gname)
			g = fid[filename_gname]
			write_h5(g, "mx", mx, staging, nmarker)
			write_h5(g, "my", my, staging, nmarker)
			write_h5(g, "mT", mT, staging, nmarker)
			close(fid)
			end

			break;
		end
		@printf("\nTotal time: ")
	end #for_time

	fid = open("eruptions.bin", "w");
	write(fid, iSample)
	write(fid, eruptionSteps)
	close(fid)

end

main()
