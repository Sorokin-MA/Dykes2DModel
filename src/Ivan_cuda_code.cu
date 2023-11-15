#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <hdf5.h>

#include <assert.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <unordered_map>

// constants
const char *bar1 = "\xc3\xc4\xc4";
const char *bar2 = "\xb3  \xc3\xc4\xc4";

// array sizes
#define SIZE_1D(n, type) ((n) * sizeof(type))
#define SIZE_2D(nx, ny, type) ((nx) * (ny) * sizeof(type))
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// CUDA-related macros
#define CUCHECK(f)                                                                                                                                             \
  do {                                                                                                                                                         \
    cudaError_t err = (f);                                                                                                                                     \
    if (err != cudaSuccess) {                                                                                                                                  \
      fprintf(stderr, "Error: cuda call '%s' finished with error: '%s'\n", #f, cudaGetErrorString(err));                                                       \
      assert(err == cudaSuccess);                                                                                                                              \
    }                                                                                                                                                          \
  } while (0)

#define CUARRAY(name, size, type)                                                                                                                              \
  type *name;                                                                                                                                                  \
  CUCHECK(cudaMalloc((void **)&name, size));                                                                                                                   \
  CUCHECK(cudaMemset(name, 0, size));

#define CUARRAY_1D(name, n, type) CUARRAY(name, SIZE_1D(n, type), type)
#define CUARRAY_2D(name, nx, ny, type) CUARRAY(name, SIZE_2D(nx, ny, type), type)

#define CUINDICES                                                                                                                                              \
  int ix = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                              \
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

#define read_h5_ds(fid, arr, name, size, buf)                                                                                                                  \
  do {                                                                                                                                                         \
    hid_t dset = H5Dopen(fid, name, H5P_DEFAULT);                                                                                                              \
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);                                                                                      \
    CUCHECK(cudaMemcpy(arr, buf, size, cudaMemcpyHostToDevice));                                                                                               \
    H5Dclose(dset);                                                                                                                                            \
  } while (0)

#define read_h5(fid, arr, size, buf) read_h5_ds(fid, arr, #arr, size, buf)

#define read_bin(fid, name, size, buf)                                                                                                                         \
  do {                                                                                                                                                         \
    fread(buf, size, 1, fid);                                                                                                                                  \
    CUCHECK(cudaMemcpy(name, buf, size, cudaMemcpyHostToDevice));                                                                                              \
  } while (0)

#define write_bin(fid, name, size, buf)                                                                                                                        \
  do {                                                                                                                                                         \
    CUCHECK(cudaMemcpy(buf, name, size, cudaMemcpyDeviceToHost));                                                                                              \
    fwrite(buf, size, 1, fid);                                                                                                                                 \
  } while (0)

#define H5Z_FILTER_BLOSC 32001

#define write_h5_ds(fid, arr, name, size, type, dims, chunks, buf)                                                                                             \
  do {                                                                                                                                                         \
    hid_t space = H5Screate_simple(ARRAY_SIZE(dims), dims, NULL);                                                                                              \
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);                                                                                                                \
    H5Pset_filter(dcpl, H5Z_FILTER_BLOSC, H5Z_FLAG_OPTIONAL, 0, NULL);                                                                                         \
    H5Pset_chunk(dcpl, ARRAY_SIZE(chunks), chunks);                                                                                                            \
    hid_t dset = H5Dcreate(fid, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);                                                                            \
    CUCHECK(cudaMemcpy(buf, arr, size, cudaMemcpyDeviceToHost));                                                                                               \
    H5Dwrite(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);                                                                                                  \
    H5Pclose(dcpl);                                                                                                                                            \
    H5Dclose(dset);                                                                                                                                            \
    H5Sclose(space);                                                                                                                                           \
  } while (0)

#define write_h5d(fid, arr, size, dims, chunks, buf) write_h5_ds(fid, arr, #arr, size, H5T_NATIVE_DOUBLE, dims, chunks, buf)

#define write_h5i(fid, arr, size, dims, chunks, buf) write_h5_ds(fid, arr, #arr, size, H5T_NATIVE_INT, dims, chunks, buf)

// array indexing
#define idc(ix, iy) ((iy)*nx + (ix))
#define idx(ix, iy) ((iy) * (nx + 1) + (ix))
#define idy(ix, iy) ((iy)*nx + (ix))
#define idxy(ix, iy) ((iy) * (nx + 1) + (ix))

// boundary conditions
#define wrapx(ix) ((ix) >= 0 ? (ix) <= nx - 1 ? (ix) : 0 : nx - 1)
#define wrapy(iy) ((iy) >= 0 ? (iy) <= ny - 1 ? (iy) : 0 : ny - 1)

#define expx(ix) ((ix) >= 0 ? (ix) <= nx - 1 ? (ix) : nx - 1 : 0)
#define expy(iy) ((iy) >= 0 ? (iy) <= ny - 1 ? (iy) : ny - 1 : 0)

#if !defined(bcx)
#define bcx expx
#endif

#if !defined(bcy)
#define bcy expy
#endif

#if !defined(GPU_ID)
#define GPU_ID 0
#endif

#if !defined(NDIGITS)
#define NDIGITS 4
#endif

#define bc_idc(ix, iy) idc(bcx(ix), bcy(iy))

#define read_par(par_name, par)                                                                                                                                \
  par_name = par[ipar];                                                                                                                                        \
  ipar++

#define ALL_PARAMS                                                                                                                                             \
  double *T, double *T_old, double *C, double *wts, double *px, double *py, double *pT, int8_t *pPh, double lam_r_rhoCp, double lam_m_rhoCp, double L_Cp,      \
      double T_top, double T_bot, double dx, double dy, double dt, double pic_amount, int nx, int ny, int npartcl, int npartcl0

#define ALL_ARGS T, T_old, C, wts, px, py, pT, pPh, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, npartcl, npartcl0

#if !defined(dmf_magma)
#define dmf_magma dmf_rhyolite
#endif

#if !defined(dmf_rock)
#define dmf_rock dmf_rhyolite
#endif

#if !defined(mf_magma)
#define mf_magma mf_rhyolite
#endif

#if !defined(mf_rock)
#define mf_rock mf_rhyolite
#endif

// kernels

__forceinline__ __device__ double dmf_rhyolite(double T) {
  double t1 = T * T;
  double t9 = exp(0.961026e3 - 0.186618e-5 * t1 * T + t1 * 0.447948e-2 + T * (-0.359050e1));
  double t12 = (0.1e1 + t9) * (0.1e1 + t9);
  return 0.559856e-5 / t12 * t9 * (t1 - 0.160022e4 * T + 0.641326e6);
}

__forceinline__ __device__ double dmf_basalt(double T) {
  double t1 = T * T;
  double t11 = exp(0.143636887899999948e3 - 0.2214446257e-6 * t1 * T + t1 * 0.572468110399999928e-3 + T * (-0.494427718499999891e0));
  double t14 = pow(0.1e1 + t11, 0.2e1);
  return 0.6643338771e-6 * (t1 - 0.1723434948e4 * T + 0.7442458310e6) * t11 / t14;
}

__forceinline__ __device__ double mf_rhyolite(double T) {
  double t2 = T * T;
  double t7 = exp(0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 - 0.1866187556e-5 * t2 * T);
  return 0.1e1 / (0.1e1 + t7);
}

__forceinline__ __device__ double mf_basalt(double T) {
  //  T=T/1000;
  //  t2 = T * T;
  //  t7 = exp(143.636887935970 - 494.427718497039*T + 572.468110446565*t2 - 221.444625682461*t2*T);
  //  return 0.1e1/(0.1e1 + t7);
  return 0.0;
}

__global__ void update_T(ALL_PARAMS) {
  CUINDICES

  if (ix > nx - 1 || iy > ny - 1) {
    return;
  }

  double qxw, qxe, qys, qyn;

  if (ix == 0) {
    qxw = 0.0;
  } else {
    qxw = -(T[idc(ix, iy)] - T[idc(ix - 1, iy)]) / dx;
  }

  if (ix == nx - 1) {
    qxe = 0.0;
  } else {
    qxe = -(T[idc(ix + 1, iy)] - T[idc(ix, iy)]) / dx;
  }

  if (iy == 0) {
    qys = -2.0 * (T[idc(ix, iy)] - T_bot) / dy;
  } else {
    qys = -(T[idc(ix, iy)] - T[idc(ix, iy - 1)]) / dy;
  }

  if (iy == ny - 1) {
    qyn = -2.0 * (T_top - T[idc(ix, iy)]) / dy;
  } else {
    qyn = -(T[idc(ix, iy + 1)] - T[idc(ix, iy)]) / dy;
  }

  double dmf = dmf_magma(T[idc(ix, iy)]) * C[idc(ix, iy)] + dmf_rock(T[idc(ix, iy)]) * (1.0 - C[idc(ix, iy)]);
  double lam_rhoCp = (lam_m_rhoCp * C[idc(ix, iy)]) + lam_r_rhoCp * (1.0 - C[idc(ix, iy)]);

  double chi = lam_rhoCp / (1.0 + L_Cp * dmf);

  T[idc(ix, iy)] += -dt * chi * ((qxe - qxw) / dx + (qyn - qys) / dy);
}

__global__ void init_particles_T(double *pT, double T_magma, int npartcl) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  pT[ip] = T_magma;
}

__global__ void init_particles_Ph(int8_t *pPh, int8_t ph, int npartcl) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  pPh[ip] = ph;
}

template <typename T> __forceinline__ __device__ int sign(T val) { return (T(0) < val) - (val < T(0)); }

__forceinline__ __device__ double2 cart2ellipt(double f, double x, double y) {
  double2 xi_eta;
  xi_eta.x = acosh(fmax(0.5 / f * (sqrt((x + f) * (x + f) + y * y) + sqrt((x - f) * (x - f) + y * y)), 1.0));
  xi_eta.y = acos(fmin(fmax(x / (f * cosh(xi_eta.x)), -1.0), 1.0)) * sign(y);
  return xi_eta;
}

__forceinline__ __device__ double2 rot2d(double x, double y, double sb, double cb) {
  double2 x_y;
  x_y.x = x * cb - y * sb;
  x_y.y = x * sb + y * cb;
  return x_y;
}

__forceinline__ __device__ double2 crack_params(double a, double b, double nu, double G) {
  double2 p_a0;
  double f = 2 * nu * (a + b) - 2 * a - b;
  p_a0.x = -2 * b * G / f;
  p_a0.y = 0.5 * f / (nu - 1);
  return p_a0;
}

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

__global__ void p2g_project(ALL_PARAMS) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  double pxi = px[ip] / dx;
  double pyi = py[ip] / dy;

  if (pxi < -1 || pxi > nx || pyi < -1 || pyi > ny) {
    return;
  }

  int ix1 = min(max(int(pxi), 0), nx - 2);
  int iy1 = min(max(int(pyi), 0), ny - 2);
  int ix2 = ix1 + 1;
  int iy2 = iy1 + 1;

  double k11 = fmax(1 - fabs(pxi - ix1), 0.0) * fmax(1 - fabs(pyi - iy1), 0.0);
  double k12 = fmax(1 - fabs(pxi - ix1), 0.0) * fmax(1 - fabs(pyi - iy2), 0.0);
  double k21 = fmax(1 - fabs(pxi - ix2), 0.0) * fmax(1 - fabs(pyi - iy1), 0.0);
  double k22 = fmax(1 - fabs(pxi - ix2), 0.0) * fmax(1 - fabs(pyi - iy2), 0.0);

  atomicAdd(&T[idc(ix1, iy1)], k11 * pT[ip]);
  atomicAdd(&T[idc(ix1, iy2)], k12 * pT[ip]);
  atomicAdd(&T[idc(ix2, iy1)], k21 * pT[ip]);
  atomicAdd(&T[idc(ix2, iy2)], k22 * pT[ip]);

  double pC = (pPh == NULL) ? (ip > npartcl0 ? 1.0 : 0.0) : double(pPh[ip]);

  atomicAdd(&C[idc(ix1, iy1)], k11 * pC);
  atomicAdd(&C[idc(ix1, iy2)], k12 * pC);
  atomicAdd(&C[idc(ix2, iy1)], k21 * pC);
  atomicAdd(&C[idc(ix2, iy2)], k22 * pC);

  atomicAdd(&wts[idc(ix1, iy1)], k11);
  atomicAdd(&wts[idc(ix1, iy2)], k12);
  atomicAdd(&wts[idc(ix2, iy1)], k21);
  atomicAdd(&wts[idc(ix2, iy2)], k22);
}

__global__ void p2g_weight(ALL_PARAMS) {
  CUINDICES

  if (ix > nx - 1 || iy > ny - 1) {
    return;
  }

  if (wts[idc(ix, iy)] == 0.0) {
    return;
  }

  T[idc(ix, iy)] /= wts[idc(ix, iy)];
  C[idc(ix, iy)] /= wts[idc(ix, iy)];
}

__forceinline__ __device__ double blerp(double x1, double x2, double y1, double y2, double f11, double f12, double f21, double f22, double x, double y) {
  double invDxDy = 1.0 / ((x2 - x1) * (y2 - y1));

  double dx1 = x - x1;
  double dx2 = x2 - x;

  double dy1 = y - y1;
  double dy2 = y2 - y;

  return invDxDy * (f11 * dx2 * dy2 + f12 * dx2 * dy1 + f21 * dx1 * dy2 + f22 * dx1 * dy1);
}

__global__ void g2p(ALL_PARAMS) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  double pxi = px[ip] / dx;
  double pyi = py[ip] / dy;

  int ix1 = min(max(int(pxi), 0), nx - 2);
  int iy1 = min(max(int(pyi), 0), ny - 2);
  int ix2 = ix1 + 1;
  int iy2 = iy1 + 1;

  double x1 = float(ix1) * dx;
  double x2 = float(ix2) * dx;
  double y1 = float(iy1) * dy;
  double y2 = float(iy2) * dy;

  double T_pic = blerp(x1, x2, y1, y2, T[idc(ix1, iy1)], T[idc(ix1, iy2)], T[idc(ix2, iy1)], T[idc(ix2, iy2)], px[ip], py[ip]);
  double T_flip =
      pT[ip] + T_pic - blerp(x1, x2, y1, y2, T_old[idc(ix1, iy1)], T_old[idc(ix1, iy2)], T_old[idc(ix2, iy1)], T_old[idc(ix2, iy2)], px[ip], py[ip]);
  pT[ip] = T_pic * pic_amount + T_flip * (1.0 - pic_amount);
}

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

__global__ void cwLabel(int *L, int nx, int ny) {
  int iy = blockIdx.x * blockDim.x + threadIdx.x;

  if (iy > ny - 1) {
    return;
  }

  for (int ix = nx - 2; ix >= 0; --ix) {
    if (L[idc(ix, iy)] >= 0 && L[idc(ix + 1, iy)] >= 0) {
      L[idc(ix, iy)] = L[idc(ix + 1, iy)];
    }
  }
}

__device__ int find_root(int *L, int idx) {
  int label = idx;
  while (L[label] != label) {
    label = L[label];
  }
  return label;
}

__global__ void merge_labels(int *L, int div, int nx, int ny) {
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  iy = div / 2 + iy * div - 1;
  if (iy > ny - 2) {
    return;
  }

  for (int ix = 0; ix < nx; ++ix) {
    if (L[idc(ix, iy)] >= 0 && L[idc(ix, iy + 1)] >= 0) {
      int lroot = find_root(L, idc(ix, iy));
      int rroot = find_root(L, idc(ix, iy + 1));
      L[min(lroot, rroot)] = L[max(lroot, rroot)];
    }
  }
}

__global__ void relabel(int *L, int nx, int ny) {
  CUINDICES

  if (ix > nx - 1 || iy > ny - 1) {
    return;
  }

  if (L[idc(ix, iy)] >= 0) {
    L[idc(ix, iy)] = find_root(L, idc(ix, iy));
  }
}

__global__ void advect_particles_eruption(double *px, double *py, int *idx, double gamma, double dxl, double dyl, int npartcl, int ncells, int nxl, int nyl) {
  int ip = blockIdx.x * blockDim.x + threadIdx.x;

  if (ip > npartcl - 1) {
    return;
  }

  double u = 0.0, v = 0.0;

  for (int i = 0; i < ncells; ++i) {
    int ic = idx[i];
    int icx = ic % nxl;
    int icy = ic / nxl;

    double xl = icx * dxl;
    double yl = icy * dyl;

    double dxl2 = dxl * dxl;
    double dyl2 = dyl * dyl;

    double delx = px[ip] - xl;
    double dely = py[ip] - yl;
    double r = max(sqrt(delx * delx + dely * dely), sqrt(dxl2 + dyl2));
    double r2_2pi = r * r * 2 * M_PI;

    u -= dxl2 * (1.0 - gamma) * delx / r2_2pi;
    v -= dyl2 * (1.0 - gamma) * dely / r2_2pi;
  }

  px[ip] += u;
  py[ip] += v;
}

__global__ void average(double *mfl, double *T, double *C, int nl, int nx, int ny) {
  int ixl = blockIdx.x * blockDim.x + threadIdx.x;
  int iyl = blockIdx.y * blockDim.y + threadIdx.y;

  if (ixl > (nx / nl - 1) || iyl > (ny / nl - 1)) {
    return;
  }

  double avg = 0.0;
  for (int ix = ixl * nl; ix < (ixl + 1) * nl; ++ix) {
    if (ix > nx - 1) {
      break;
    }
    for (int iy = iyl * nl; iy < (iyl + 1) * nl; ++iy) {
      if (iy > ny - 1) {
        break;
      }
      double vf = C[idc(ix, iy)];
      avg += mf_magma(T[idc(ix, iy)]) * vf + mf_rock(T[idc(ix, iy)]) * (1 - vf);
    }
  }
  avg /= nl * nl;
  mfl[iyl * (nx / nl) + ixl] = avg;
}

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

__global__ void inject_particles(double *px, double *py, double *pT, int8_t *pPh, int *npartcl, const int *pcnt, const double *T, const double *C, double dx,
                                 double dy, int nx, int ny, int min_pcount, int max_npartcl) {
  CUINDICES
  if (ix > nx - 2 || iy > ny - 2) {
    return;
  }

  if (pcnt[idc(ix, iy)] < min_pcount) {
    for (int ioy = 0; ioy <= 1; ++ioy) {
      for (int iox = 0; iox <= 1; ++iox) {
        int inx = ix + iox;
        int iny = iy + ioy;
        int new_npartcl = atomicAdd(npartcl, 1);
        if (new_npartcl > max_npartcl - 1)
          break;
        int ip = new_npartcl;
        px[ip] = dx * inx;
        py[ip] = dy * iny;

        pT[ip] = T[idc(inx, iny)];
        pPh[ip] = C[idc(inx, iny)] < 0.5 ? 0 : 1;
      }
    }
  }
}

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

static void printDeviceProperties(int deviceId);

static auto tic() { return std::chrono::high_resolution_clock::now(); }

template <typename T> static void toc(const T &start) {
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  printf("%lf s\n", diff.count());
}

int main() {
  if (H5Zfilter_avail(H5Z_FILTER_BLOSC) <= 0) {
    fprintf(stderr, "Error: blosc filter is not available\n");
    exit(1);
  }

  char c;
  c = getchar();

  printDeviceProperties(GPU_ID);
  cudaSetDevice(GPU_ID);
  cudaDeviceReset();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  double dpa[18];
  int ipa[12];
  double Lx, Ly, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, T_magma, tsh, gamma, Ly_eruption, nu, G, dt, dx, dy, eiter, pic_amount;
  int pmlt, nx, ny, nl, nt, niter, nout, nsub, nerupt, npartcl, nmarker, nSample;
  char filename[1024];

  FILE *h = fopen("pa.bin", "rb");
  fread(dpa, sizeof(double), ARRAY_SIZE(dpa), h);
  fread(ipa, sizeof(int), ARRAY_SIZE(ipa), h);

  int ipar = 0;
  read_par(Lx, dpa);
  read_par(Ly, dpa);
  read_par(lam_r_rhoCp, dpa);
  read_par(lam_m_rhoCp, dpa);
  read_par(L_Cp, dpa);
  read_par(T_top, dpa);
  read_par(T_bot, dpa);
  read_par(T_magma, dpa);
  read_par(tsh, dpa);
  read_par(gamma, dpa);
  read_par(Ly_eruption, dpa);
  read_par(nu, dpa);
  read_par(G, dpa);
  read_par(dt, dpa);
  read_par(dx, dpa);
  read_par(dy, dpa);
  read_par(eiter, dpa);
  read_par(pic_amount, dpa);
  ipar = 0;
  read_par(pmlt, ipa);
  read_par(nx, ipa);
  read_par(ny, ipa);
  read_par(nl, ipa);
  read_par(nt, ipa);
  read_par(niter, ipa);
  read_par(nout, ipa);
  read_par(nsub, ipa);
  read_par(nerupt, ipa);
  read_par(npartcl, ipa);
  read_par(nmarker, ipa);
  read_par(nSample, ipa);

  double *critVol = (double *)malloc(nSample * sizeof(double));
  fread(critVol, sizeof(double), nSample, h);

  int *ndikes = (int *)malloc(nt * sizeof(int));
  fread(ndikes, sizeof(int), nt, h);

  int ndikes_all = 0;
  for (int istep = 0; istep < nt; ++istep) {
    ndikes_all += ndikes[istep];
  }

  int *particle_edges = (int *)malloc((ndikes_all + 1) * sizeof(int));
  fread(particle_edges, sizeof(int), ndikes_all + 1, h);

  int *marker_edges = (int *)malloc((ndikes_all + 1) * sizeof(int));
  fread(marker_edges, sizeof(int), ndikes_all + 1, h);

  fclose(h);

  double cap_frac = 1.5;

  int npartcl0 = npartcl;
  int max_npartcl = int(npartcl * cap_frac) + particle_edges[ndikes_all];

  int nmarker0 = nmarker;
  int max_nmarker = nmarker + marker_edges[ndikes_all];

  dim3 blockSize(16, 32);
  dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

  CUARRAY_2D(T, nx, ny, double)
  CUARRAY_2D(T_old, nx, ny, double)
  CUARRAY_2D(C, nx, ny, double)
  CUARRAY_2D(wts, nx, ny, double)
  CUARRAY_2D(pcnt, nx, ny, int)

  CUARRAY_1D(px, max_npartcl, double)
  CUARRAY_1D(py, max_npartcl, double)
  CUARRAY_1D(pT, max_npartcl, double)
  CUARRAY_1D(pPh, max_npartcl, int8_t)

  int np_dikes = particle_edges[ndikes_all];
  CUARRAY_1D(px_dikes, np_dikes, double);
  CUARRAY_1D(py_dikes, np_dikes, double);

  CUARRAY_1D(mx, max_nmarker, double)
  CUARRAY_1D(my, max_nmarker, double)
  CUARRAY_1D(mT, max_nmarker, double)

  void *staging = nullptr;
  CUCHECK(cudaMallocHost(&staging, SIZE_1D(max_npartcl, double)));

  int *npartcl_d = nullptr;
  CUCHECK(cudaMallocHost(&npartcl_d, sizeof(int)));

  int nxl = nx / nl;
  int nyl = ny / nl;

  int *L;
  cudaMalloc(&L, SIZE_2D(nxl, nyl, int));
  int *L_host = (int *)malloc(SIZE_2D(nxl, nyl, int));

  CUARRAY_2D(mfl, nxl, nyl, double);

  double *dike_a = (double *)malloc(ndikes_all * sizeof(double));
  double *dike_b = (double *)malloc(ndikes_all * sizeof(double));
  double *dike_x = (double *)malloc(ndikes_all * sizeof(double));
  double *dike_y = (double *)malloc(ndikes_all * sizeof(double));
  double *dike_t = (double *)malloc(ndikes_all * sizeof(double));

  h = fopen("dikes.bin", "rb");

  fread(dike_a, sizeof(double), ndikes_all, h);
  fread(dike_b, sizeof(double), ndikes_all, h);
  fread(dike_x, sizeof(double), ndikes_all, h);
  fread(dike_y, sizeof(double), ndikes_all, h);
  fread(dike_t, sizeof(double), ndikes_all, h);

  fclose(h);

  hid_t fid = H5Fopen("particles.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  read_h5(fid, px, SIZE_1D(npartcl, double), staging);
  read_h5(fid, py, SIZE_1D(npartcl, double), staging);

  read_h5(fid, px_dikes, SIZE_1D(np_dikes, double), staging);
  read_h5(fid, py_dikes, SIZE_1D(np_dikes, double), staging);
  H5Fclose(fid);

  fid = H5Fopen("markers.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t gid = H5Gopen(fid, "0", H5P_DEFAULT);
  read_h5(gid, mx, SIZE_1D(max_nmarker, double), staging);
  read_h5(gid, my, SIZE_1D(max_nmarker, double), staging);
  read_h5(gid, mT, SIZE_1D(max_nmarker, double), staging);
  H5Gclose(gid);
  H5Fclose(fid);

  sprintf(filename, "grid.%0*d.h5", NDIGITS, 0);
  fid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  read_h5(fid, T, SIZE_2D(nx, ny, double), staging);
  read_h5(fid, C, SIZE_2D(nx, ny, double), staging);
  H5Fclose(fid);

  auto tm_all = tic();

  // init
  printf("%s initialization              \xb3 ", bar1);
  auto tm = tic();
  double pic_amount_tmp = pic_amount;
  pic_amount = 1.0;
  dim3 blockSize1D(768);
  dim3 gridSize1D((npartcl + blockSize1D.x - 1) / blockSize1D.x);
  g2p<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
  gridSize1D.x = (max_npartcl - npartcl + blockSize1D.x - 1) / blockSize1D.x;
  init_particles_T<<<gridSize1D, blockSize1D>>>(pT + npartcl, T_magma, max_npartcl - npartcl);
  init_particles_Ph<<<gridSize1D, blockSize1D>>>(pPh + npartcl, 1, max_npartcl - npartcl);
  gridSize1D.x = (max_nmarker - nmarker + blockSize1D.x - 1) / blockSize1D.x;
  init_particles_T<<<gridSize1D, blockSize1D>>>(mT + nmarker, T_magma, max_nmarker - nmarker);
  CUCHECK(cudaDeviceSynchronize());
  pic_amount = pic_amount_tmp;
  toc(tm);

  int idike = 0;
  int iSample = 0;
  std::vector<int> eruptionSteps;

  // action
  for (int it = 1; it <= nt; ++it) {
    printf("%s it = %d\n", bar1, it);

    bool is_eruption = false;
    bool is_intrusion = (ndikes[it - 1] > 0);

    if (it % nerupt == 0) {
      printf("%s checking melt fraction   \xb3 ", bar2);
      auto tm = tic();

      dim3 blockSizel(16, 32);
      dim3 gridSizel((nxl + blockSizel.x - 1) / blockSizel.x, (nyl + blockSizel.y - 1) / blockSizel.y);
      average<<<gridSizel, blockSizel>>>(mfl, T, C, nl, nx, ny);
      CUCHECK(cudaDeviceSynchronize());
      ccl(mfl, L, tsh, nxl, nyl);
      cudaMemcpy(L_host, L, SIZE_2D(nxl, nyl, int), cudaMemcpyDeviceToHost);

      std::unordered_map<int, int> volumes;

      for (int iy = 0; iy < nyl; ++iy) {
        if (iy * dy * nl < Ly_eruption) {
          continue;
        }
        for (int ix = 0; ix < nxl; ++ix) {
          if (L_host[iy * nxl + ix] >= 0) {
            volumes[L_host[iy * nxl + ix]]++;
          }
        }
      }

      int maxVol = -1;
      int maxIdx = -1;

      for (const auto &[idx, vol] : volumes) {
        if (vol > maxVol) {
          maxVol = vol;
          maxIdx = idx;
        }
      }
      toc(tm);

      double dxl = dx * nl;
      double dyl = dy * nl;

      if (maxVol * dxl * dyl >= critVol[iSample]) {
        printf("%s erupting %07d cells   \xb3 ", bar2, maxVol);
        auto tm = tic();
        int *cell_idx;
        cudaMalloc(&cell_idx, SIZE_1D(maxVol, int));
        int *cell_idx_host = (int *)malloc(SIZE_1D(maxVol, int));

        int next_idx = 0;
        for (int idx = 0; idx < nxl * nyl; ++idx) {
          if (L_host[idx] == maxIdx) {
            cell_idx_host[next_idx++] = idx;
          }
        }

        cudaMemcpy(cell_idx, cell_idx_host, SIZE_1D(maxVol, int), cudaMemcpyHostToDevice);

        blockSize1D.x = 512;
        gridSize1D.x = (npartcl + blockSize1D.x - 1) / blockSize1D.x;
        advect_particles_eruption<<<gridSize1D, blockSize1D>>>(px, py, cell_idx, gamma, dxl, dyl, npartcl, maxVol, nxl, nyl);
        CUCHECK(cudaDeviceSynchronize());
        gridSize1D.x = (nmarker + blockSize1D.x - 1) / blockSize1D.x;
        advect_particles_eruption<<<gridSize1D, blockSize1D>>>(mx, my, cell_idx, gamma, dxl, dyl, nmarker, maxVol, nxl, nyl);
        CUCHECK(cudaDeviceSynchronize());

        cudaFree(cell_idx);
        free(cell_idx_host);

        iSample++;

        is_eruption = true;
        eruptionSteps.push_back(it);

        toc(tm);
      }
    }

    if (is_intrusion) {
      printf("%s inserting %02d dikes       \xb3 ", bar2, ndikes[it - 1]);
      auto tm = tic();

      for (int i = 0; i < ndikes[it - 1]; ++i, ++idike) {
        blockSize1D.x = 512;
        // advect particles
        gridSize1D.x = (npartcl + blockSize1D.x - 1) / blockSize1D.x;
        advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(px, py, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
                                                                ndikes[it - 1], npartcl);
        int dike_start = particle_edges[idike];
        int dike_end = particle_edges[idike + 1];
        int np_dike = dike_end - dike_start;

        if (npartcl + np_dike > max_npartcl) {
          fprintf(stderr, "ERROR: number of particles exceeds maximum value, increase capacity\n");
          exit(EXIT_FAILURE);
        }

        cudaMemcpy(&px[npartcl], &px_dikes[dike_start], np_dike * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&py[npartcl], &py_dikes[dike_start], np_dike * sizeof(double), cudaMemcpyDeviceToDevice);
        npartcl += np_dike;
        // advect markers
        gridSize1D.x = (nmarker + blockSize1D.x - 1) / blockSize1D.x;
        advect_particles_intrusion<<<gridSize1D, blockSize1D>>>(mx, my, dike_a[idike], dike_b[idike], dike_x[idike], dike_y[idike], dike_t[idike], nu, G,
                                                                ndikes[it - 1], nmarker);
        CUCHECK(cudaDeviceSynchronize());
        nmarker += marker_edges[idike + 1] - marker_edges[idike];
      }
      toc(tm);
    }

    if (is_eruption || is_intrusion) {
      printf("%s p2g interpolation        \xb3 ", bar2);
      tm = tic();

      cudaMemset(T, 0, SIZE_2D(nx, ny, double));
      cudaMemset(C, 0, SIZE_2D(nx, ny, double));
      cudaMemset(wts, 0, SIZE_2D(nx, ny, double));

      blockSize1D.x = 768;
      gridSize1D.x = (npartcl + blockSize1D.x - 1) / blockSize1D.x;
      p2g_project<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
      CUCHECK(cudaDeviceSynchronize());
      p2g_weight<<<gridSize, blockSize>>>(ALL_ARGS);
      CUCHECK(cudaDeviceSynchronize());
      toc(tm);

      printf("%s particle injection ", bar2);
      tm = tic();
      blockSize1D.x = 512;
      gridSize1D.x = (npartcl + blockSize1D.x - 1) / blockSize1D.x;
      cudaMemset(pcnt, 0, SIZE_2D(nx, ny, int));
      count_particles<<<gridSize1D, blockSize1D>>>(pcnt, px, py, dx, dy, nx, ny, npartcl);
      CUCHECK(cudaDeviceSynchronize());

      cudaMemcpy(npartcl_d, &npartcl, sizeof(int), cudaMemcpyHostToDevice);
      int min_pcount = 2;
      inject_particles<<<gridSize, blockSize>>>(px, py, pT, pPh, npartcl_d, pcnt, T, C, dx, dy, nx, ny, min_pcount, max_npartcl);
      CUCHECK(cudaDeviceSynchronize());
      int new_npartcl = npartcl;
      cudaMemcpy(&new_npartcl, npartcl_d, sizeof(int), cudaMemcpyDeviceToHost);
      if (new_npartcl > max_npartcl) {
        fprintf(stderr, "ERROR: number of particles exceeds maximum value, increase capacity\n");
        exit(EXIT_FAILURE);
      }
      if (new_npartcl > npartcl) {
        printf("(%03d) \xb3 ", new_npartcl - npartcl);
        npartcl = new_npartcl;
      } else {
        printf("(000) \xb3 ", bar2);
      }
      toc(tm);
    }

    printf("%s solving heat diffusion   \xb3 ", bar2);
    tm = tic();
    CUCHECK(cudaMemcpy(T_old, T, SIZE_2D(nx, ny, double), cudaMemcpyDeviceToDevice));
    for (int isub = 0; isub < nsub; ++isub) {
      update_T<<<gridSize, blockSize>>>(ALL_ARGS);
      CUCHECK(cudaDeviceSynchronize());
    }
    toc(tm);

    printf("%s g2p interpolation        \xb3 ", bar2);
    tm = tic();
    // particles g2p
    blockSize1D.x = 1024;
    gridSize1D.x = (npartcl + blockSize1D.x - 1) / blockSize1D.x;
    g2p<<<gridSize1D, blockSize1D>>>(ALL_ARGS);
    // markers g2p
    gridSize1D.x = (nmarker + blockSize1D.x - 1) / blockSize1D.x;
    double pic_amount_tmp = pic_amount;
    pic_amount = 1.0;
    g2p<<<gridSize1D, blockSize1D>>>(T, T_old, C, wts, mx, my, mT, NULL, lam_r_rhoCp, lam_m_rhoCp, L_Cp, T_top, T_bot, dx, dy, dt, pic_amount, nx, ny, nmarker,
                                     nmarker0);
    CUCHECK(cudaDeviceSynchronize());
    pic_amount = pic_amount_tmp;
    toc(tm);

    if (it % nout == 0 || is_eruption) {
      auto tm = tic();
      printf("%s writing results to disk  \xb3 ", bar2);
      sprintf(filename, "grid.%0*d.h5", NDIGITS, it);
      hsize_t dims[] = {(hsize_t)ny, (hsize_t)nx};
      hsize_t chunks[] = {(hsize_t)ny, (hsize_t)nx};
      fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      write_h5d(fid, T, SIZE_2D(nx, ny, double), dims, chunks, staging);
      write_h5d(fid, C, SIZE_2D(nx, ny, double), dims, chunks, staging);
      if (is_eruption) {
        hsize_t dims[] = {(hsize_t)nyl, (hsize_t)nxl};
        hsize_t chunks[] = {(hsize_t)nyl, (hsize_t)nxl};
        write_h5i(fid, L, SIZE_2D(nxl, nyl, int), dims, chunks, staging);
      }
      H5Fclose(fid);
      toc(tm);
    }

    auto tm = tic();
    printf("%s writing markers to disk  \xb3 ", bar2);
    fid = H5Fopen("markers.h5", H5F_ACC_RDWR, H5P_DEFAULT);
    hsize_t dims1d[] = {(hsize_t)nmarker};
    hsize_t chunks1d[] = {(hsize_t)nmarker};
    char _gname[1024];
    sprintf(_gname, "%d", it);
    hid_t gid = H5Gcreate(fid, _gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_h5d(gid, mx, SIZE_1D(nmarker, double), dims1d, chunks1d, staging);
    write_h5d(gid, my, SIZE_1D(nmarker, double), dims1d, chunks1d, staging);
    write_h5d(gid, mT, SIZE_1D(nmarker, double), dims1d, chunks1d, staging);
    H5Gclose(gid);
    H5Fclose(fid);
    toc(tm);
  }

  printf("Total time: ");
  toc(tm_all);

  h = fopen("eruptions.bin", "wb");
  fwrite(&iSample, sizeof(int), 1, h);
  fwrite(eruptionSteps.data(), sizeof(int), iSample, h);
  fclose(h);

  cudaFree(T);
  cudaFree(T_old);
  cudaFree(C);
  cudaFree(px);
  cudaFree(py);
  cudaFree(pT);
  cudaFree(mx);
  cudaFree(my);
  cudaFree(mT);
  cudaFree(mfl);
  cudaFree(L);
  cudaFreeHost(staging);
  cudaFree(npartcl_d);

  free(critVol);
  free(L_host);
  free(dike_a);
  free(dike_b);
  free(dike_x);
  free(dike_y);
  free(dike_t);
  free(particle_edges);
  free(marker_edges);

  return EXIT_SUCCESS;
}

void printDeviceProperties(int deviceId) {
  cudaDeviceProp deviceProperties;
  CUCHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));

  printf("Device:\n");
  printf("    id: %d\n", deviceId);
  printf("  name: %s\n\n", deviceProperties.name);
}
