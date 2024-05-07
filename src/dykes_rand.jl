include("dykes_init.jl")
include("dykes_structs.jl")

function dikes_rand()
	Random.seed!(1234)


	gpuid      = 0;			#gpu id
	tyear = 365*24*3600;	#seconds in year

Lx          = 20000; # x size of area, m
Ly          = 25000; # y size of area, m %20000
Lx_Ly       = Lx/Ly;
narrow_fact = 0.6;
dike_x_W    = 5000; #m 
dike_x_Wn    = dike_x_W*narrow_fact; #m 
dike_a_rng  = [100 1500]; #m
dike_b_rng  = [10 20]; #m

dike_x_rng  = [ (Lx-dike_x_W)/2 (Lx+dike_x_W)/2] ;
dike_x_rng_n  = [ (Lx-dike_x_Wn)/2 (Lx+dike_x_Wn)/2] ;

dike_y_rng  = [1000 22000];				#dikes y distribution
dike_t_rng  = [0.95*pi/2 1.05*pi/2];	#dykes time distribution
dike_to_sill = 21000;					#boundary where dykes turn yourself to sill, m
dz           = 5000;					#z dimension? i guess, m

Lam_r       = 1.5;			#thermal conductivity of rock, W/m/K
Lam_m       = 1.2;			#thermal conductivity of magma, W/m/K
rho         = 2650;			#density, kg/m^3
Cp          = 1350;			#scpecifiv heat capacity, J/kg/K
Lheat       = 3.5e5;		#Latent heat of melting, J/kg
T_top       = 100;			#temperature at depth 5 km, C
dTdy        = 20;			#how fast temperature decreasing with depth, K/km
T_magma     = 950;			#magma intrusion temperature, C
T_ch        = 700;			#?
Qv          = 0.038;		#m^3/s
dt          = 50*tyear;		#time
tfin        = 600e3*tyear;
terupt      = 600e3*tyear;


Ly_eruption = 2000; # m
lam_r_rhoCp = Lam_r/(rho*Cp); # m^2/s
dT          = 500; # K
E           = 1.56e10; # Pa
nu          = 0.3;

# scales
tsc         = Ly^2/lam_r_rhoCp; # s

# nondimensional
tsh         = 0.75;
lam_m_lam_r = Lam_m/Lam_r;
gamma       = 0.1;

Ste         = dT/(Lheat/Cp); # Ste = dT/L_Cp

# dimensionally dependent
lam_m_rhoCp = lam_r_rhoCp*lam_m_lam_r;
Omx         = Lx/2 - Lx/3;
Lmx         = 2/3*Lx;
Omy         = Ly/2 - Ly/3;
Lmy         = 2/3*Ly;
L_Cp        = dT/Ste;

q           = Qv/dz;
G           = E/(2*(1+nu));

alpha       = 2; # parameter
Nsample     = 1000; #size of a sample

critVol = ones(1, 1000);
critVol_hist = [265, 50, 0.5, 0.02,	0.64, 0.02, 0.02, 0.7, 0.02, 0.001, 0.001, 0.06, 0.05, 0.02, 0.07, 0.052, 0.854, 0.026, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029]

critVol_h = @view critVol[1:24]
copy!(critVol_h, critVol_hist)
#critVol(1:24) = critVol_hist(1:end);
critVol = 10^9*critVol/dz/(1-gamma);

#numerics
steph       = 5;
ny          = fix(Ly/steph);
nx          = fix(Lx_Ly*ny);
nl          = 4;				
nmy         = 200;
nmx         = fix(Lmx/Lmy*nmy);
pmlt        = 2;
niter       = nx;
eiter       = 1e-12;
CFL         = 0.23;
pic_amount  = 0.05;
nout        = 10000;
nt          = tfin/dt;
nt_erupt    = terupt/dt;
nerupt      = 1;

#preprocessing
dx          = Lx/(nx-1);
dy          = Ly/(ny-1);
dr          = min(dx,dy)/pmlt;
dmx         = Lmx/(nmx-1);
dmy         = Lmy/(nmy-1);
dmr         = min(dmx,dmy);
xs          = 0:dx:Lx;
ys          = 0:dy:Ly;
x,y         = ndgrid(xs,ys);
nbd         = fix(0.1*(ny-1));
pxs         = -nbd*dx-dx/pmlt/2:dx/pmlt:Lx+nbd*dx+dx/pmlt-dx/pmlt/2;
pys         = -nbd*dy-dy/pmlt/2:dy/pmlt:Ly+nbd*dy+dy/pmlt-dy/pmlt/2;
[px,py]     = ndgrid(pxs,pys);
px          = px(:);
py          = py(:);
mxs         = Omx:dmx:Omx+Lmx;
mys         = Omy:dmy:Omy+Lmy;
[mx,my]     = ndgrid(mxs,mys);
mx          = mx(:);
my          = my(:);
dt_diff     = CFL*min(dx,dy)^2/lam_r_rhoCp;
nsub        = ceil(dt/dt_diff);
dt_diff     = dt/nsub;
npartcl     = numel(px);
nmarker     = numel(mx);
T_bot       = T_top + dTdy*Ly/1e3;
ndigits     = floor(log10(nt))+1;

#init
T           = T_top + dTdy*(Ly-y)/1e3;
indx=find(xs > dike_x_rng(1) & xs < dike_x_rng(2));
indy=find(ys >dike_y_rng(1)  &  ys <dike_y_rng(2));

#print T

C           = zeros(nx,ny);
	return 0
end
