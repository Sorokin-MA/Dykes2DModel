clear;
seed = 123;
rng(seed);
% program config
% cur_dir    =pwd;

sim_dir    = '.';	%directory name where will be compilled files
sim_name   = 'magma_chamber_eruption_rh_rh_particles_generation';
sim_files   = 'magma_chamber_eruption_rh_rh_particles_generation.*';
sim_driver = mfilename('fullpath');
exe_name   = [sim_name '.exe'];
exe_path   = [sim_dir '/' exe_name];
cu_name    = [sim_name '.cu'];
cu_path    = [sim_dir '/' cu_name];
cuda_arch  = 'sm_86';
%cuda_ccbin = '"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64"';
cuda_ccbin = '"c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64"';	%where cuda binaries
hdf5_path  = 'C:\Program Files\HDF_Group\HDF5\1.14.1';		%where hdf5 library
addpath([hdf5_path '\lib'], [hdf5_path '\lib\plugin'])
bcx        = 'expx';
bcy        = 'expy';
gpuid      = 0;				%gpu id
tyear = 365*24*3600;		%seconds in year

% cleanup
if ~exist(sim_dir,'dir')
    mkdir(sim_dir)
end
delete([sim_dir '/*.bin']);
delete([sim_dir '/*.h5']);

% physics
% dimensionally independent
Lx          = 20000; % x size of area, m %20000
Ly          = 20000; % y size of area, m %20000
Lx_Ly       = Lx/Ly; % Lx/Ly
narrow_fact = 1;
dyke_x_W    = 10000; %m where is x point
dyke_x_Wn    = dyke_x_W*narrow_fact; %m 
dyke_a_rng  = [100 1500]; %m length
dyke_b_rng  = [10 20]; %m width

dyke_x_rng  = [ (Lx-dyke_x_W)/2 (Lx+dyke_x_W)/2] ;		%distribution for centre of dykes
dyke_x_rng_n  = [ (Lx-dyke_x_Wn)/2 (Lx+dyke_x_Wn)/2];

dyke_y_rng  = [5000 13000];				%dykes y distribution
dyke_t_rng  = [0.95*pi/2 1.05*pi/2];	%dykes time distribution
dyke_to_sill = 6000;					%boundary where dykes turn yourself to sill, сверху, m
dz           = 10000;					%z dimension? i guess, m

Lam_r       = 1.5;						%thermal conductivity of rock, W/m/K
Lam_m       = 1.2;						%thermal conductivity of magma, W/m/K
rho         = 2650;						%density, kg/m^3
Cp          = 1350;						%scpecifiv heat capacity, J/kg/K
Lheat       = 3.5e5;					%Latent heat of melting, J/kg
T_top       = 100;						%temperature at depth 5 km, C
dTdy        = 20;						%how fast temperature decreasing with depth, K/km
T_magma     = 1050;						%magma intrusion temperature, C
T_ch        = 700;						%WARN:уточнить		%?
Qv          = 0.00411 * 1.e9 / tyear;	%приток магмы	%m^3/s
dt          = 5*tyear;					%time
tfin        = 150e3*tyear;				%final time
terupt      = 150e3*tyear;


Ly_eruption = 2000; % m
lam_r_rhoCp = Lam_r/(rho*Cp); % m^2/s
dT          = 500; % K
E           = 1.56e10; % Pa
nu          = 0.3;

% scales
tsc         = Ly^2/lam_r_rhoCp; % s

% nondimensional
tsh         = 0.75;
lam_m_lam_r = Lam_m/Lam_r;
gamma       = 0.1;

Ste         = dT/(Lheat/Cp); % Ste = dT/L_Cp

% dimensionally dependent
lam_m_rhoCp = lam_r_rhoCp*lam_m_lam_r;
Omx         = Lx/2 - Lx/3;
Lmx         = 2/3*Lx;
Omy         = Ly/2 - Ly/3;
Lmy         = 2/3*Ly;
L_Cp        = dT/Ste;

q           = Qv/dz;
G           = E/(2*(1+nu));

alpha       = 2; % parameter
Nsample     = 1000; % size of a sample

%NOTE: Заменить на считывание из файла, это объёмы
 %distr      = random('exponential',alpha,1,Nsample);
distr       = -alpha .* log(rand(1,Nsample, 'like', alpha)); % == expinv(u, mu)
rn          = (distr-min(distr))/(max(distr)-min(distr));
critVol     = 10.^(9+3*rn)        /dz/(1-gamma); %km^3
%dyke_x_rng  = [0.32 0.67]*Lx;

critVol = ones(1, 1000);
%critVol_hist = [39.8, 14.9, 14.3, 13, 12, 12.8, 11.8, 11, 11.5, 11, 10.6, 9.6, 9.3, 5.1, 4.9, 4.5, 4.3, 4.2, 4.2, 4.2, 4.1, 3.9, 0.5];
critVol_hist = [265, 50, 0.5, 0.02,	0.64, 0.02, 0.02, 0.7, 0.02, 0.001, 0.001, 0.06, 0.05, 0.02, 0.07, 0.052, 0.854, 0.026, 0.018, 0.12, 0.661, 0.016, 0.02, 0.029]

critVol(1:24) = critVol_hist(1:end);
critVol = 10^9*critVol/dz/(1-gamma);

% numerics
steph       = 5;
ny          = fix(Ly/steph);
nx          = fix(Lx_Ly*ny);
nl          = 4;					%?
nmy         = 200;
nmx         = fix(Lmx/Lmy*nmy);
pmlt        = 2;
niter       = nx;
eiter       = 1e-12;
CFL         = 0.23;
pic_amount  = 0.05;
nout        = 5000;
nt          = tfin/dt;
nt_erupt    = terupt/dt;
nerupt      = 1;

% preprocessing
dx          = Lx/(nx-1);
dy          = Ly/(ny-1);
dr          = min(dx,dy)/pmlt;
dmx         = Lmx/(nmx-1);
dmy         = Lmy/(nmy-1);
dmr         = min(dmx,dmy);
xs          = 0:dx:Lx;
ys          = 0:dy:Ly;
[x,y]       = ndgrid(xs,ys);
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

% init
T           = T_top + dTdy*(Ly-y)/1e3;
indx=find(xs > dyke_x_rng(1) & xs < dyke_x_rng(2));
indy=find(ys >dyke_y_rng(1)  &  ys <dyke_y_rng(2));

%T(indx,indy)=T_ch;
pcolor(x,y,T),shading flat,axis image;c= colorbar,drawnow

C           = zeros(nx,ny);
% generate dykes
Q      = 0;
dyke_a = [];
dyke_b = [];
dyke_x = [];
dyke_y = [];
dyke_t = [];
dyke_v = [];
Vtot=q*nt_erupt*dt;
Q_tsh       = 0.5*Vtot;
while Q < Vtot
    dyke_a = [dyke_a dyke_a_rng(1) + diff(dyke_a_rng)*rand];
    dyke_b = [dyke_b dyke_b_rng(1) + diff(dyke_b_rng)*rand];
    if Q < Q_tsh
        dyke_x = [dyke_x dyke_x_rng(1) + diff(dyke_x_rng)*rand];
    else
        dyke_x = [dyke_x dyke_x_rng_n(1) + diff(dyke_x_rng_n)*rand];
    end
    dyke_y = [dyke_y dyke_y_rng(1) + diff(dyke_y_rng)*rand];
    dyke_t = [dyke_t dyke_t_rng(1) + diff(dyke_t_rng)*rand];
    dyke_v = [dyke_v pi*dyke_a(end)*dyke_b(end)];
    Q      = Q + dyke_v(end);
end
dyke_v = [0 cumsum(dyke_v)];
dv     = dyke_v(end)/nt_erupt;
ndykes = diff(floor(interp1(dyke_v,1:numel(dyke_v),0:dv:dyke_v(end))));
assert(numel(ndykes) == nt_erupt);
ndykes(numel(ndykes)+1:nt) = 0;
assert(numel(ndykes) == nt);
dyke_npartcl = zeros(1,sum(ndykes));
dyke_nmarker = zeros(1,sum(ndykes));
px_dyke      = cell(sum(ndykes),1);
py_dyke      = cell(sum(ndykes),1);
mx_dyke      = cell(sum(ndykes),1);
my_dyke      = cell(sum(ndykes),1);
dyke_t(dyke_y>=dyke_to_sill) = dyke_t(dyke_y>=dyke_to_sill) + pi/2; %reverse dykes to sills
for idyke = 1:sum(ndykes)
    a              = dyke_a(idyke);
    b              = dyke_b(idyke);
    dykex0         = dyke_x(idyke);
    dykey0         = dyke_y(idyke);
    st             = sin(dyke_t(idyke));
    ct             = cos(dyke_t(idyke));
    % markers
    dykexs         = linspace(-a,a,round(2*a/dr));
    dykeys         = linspace(-b,b,round(2*b/dr));
    if isempty(dykexs);dykexs = 0;end
    if isempty(dykeys);dykeys = 0;end
    [dykex,dykey]  = ndgrid(dykexs,dykeys);
    dykex          = dykex(:);
    dykey          = dykey(:);
    outside        = (dykex.^2/a^2 + dykey.^2/b^2) > 1+eps;
    dykex(outside) = [];
    dykey(outside) = [];
    px_dyke{idyke} = dykex0 + dykex*ct - dykey*st;
    py_dyke{idyke} = dykey0 + dykex*st + dykey*ct;
    dyke_npartcl(idyke) = numel(px_dyke{idyke});
    % markers
    dykemxs         = linspace(-a,a,round(2*a/dmr));
    dykemys         = linspace(-b,b,round(2*b/dmr));
    if numel(dykemxs)<=1;dykemxs = 0;end
    if numel(dykemys)<=1;dykemys = 0;end
    [dykemx,dykemy]  = ndgrid(dykemxs,dykemys);
    dykemx          = dykemx(:);
    dykemy          = dykemy(:);
    outside         = (dykemx.^2/a^2 + dykemy.^2/b^2) > 1+eps;
    dykemx(outside) = [];
    dykemy(outside) = [];
    mx_dyke{idyke}  = dykex0 + dykemx*ct - dykemy*st;
    my_dyke{idyke}  = dykey0 + dykemx*st + dykemy*ct;
    dyke_nmarker(idyke) = numel(mx_dyke{idyke});
end
px_dykes     = cell2mat(px_dyke);
py_dykes     = cell2mat(py_dyke);
mx           = [mx;cell2mat(mx_dyke)];
my           = [my;cell2mat(my_dyke)];
mT           = T_top + dTdy/1e3*(Ly-my);
mT(mx > dyke_x_rng(1) & mx < dyke_x_rng(2) & my > dyke_y_rng(1)  & my <dyke_y_rng(2))=T_ch;


partcl_edges = [0 cumsum(dyke_npartcl)];
marker_edges = [0 cumsum(dyke_nmarker)];

% save data
fid        = fopen([sim_dir '/pa.bin'],'w');
fwrite(fid,[Lx Ly lam_r_rhoCp lam_m_rhoCp L_Cp T_top T_bot T_magma tsh gamma Ly_eruption nu G dt_diff dx dy eiter pic_amount],'double');
fwrite(fid,[pmlt nx ny nl nt niter nout nsub nerupt npartcl nmarker Nsample],'int32');
fwrite(fid,critVol,'double');
fwrite(fid,ndykes,'int32');
fwrite(fid,partcl_edges,'int32');
fwrite(fid,marker_edges,'int32');
fclose(fid);
fid        = fopen([sim_dir '/dykes.bin'],'w');
fwrite(fid,[dyke_a dyke_b dyke_x dyke_y dyke_t],'double');
fclose(fid);
fname = sprintf('%s/grid.%0*d.h5', sim_dir,ndigits,0);
h5create(fname,'/T',size(T),'ChunkSize',size(T),'Deflate',5);
h5create(fname,'/C',size(C),'ChunkSize',size(C),'Deflate',5);
h5write(fname,'/T',T);
h5write(fname,'/C',C);
fname = [sim_dir '/particles.h5'];
h5create(fname, '/px',size(px),'ChunkSize',size(px),'Deflate',5);
h5create(fname, '/py',size(py),'ChunkSize',size(py),'Deflate',5);
h5write(fname,'/px',px);
h5write(fname,'/py',py);
h5create(fname, '/px_dykes',size(px_dykes),'ChunkSize',size(px_dykes),'Deflate',5);
h5create(fname, '/py_dykes',size(py_dykes),'ChunkSize',size(py_dykes),'Deflate',5);
h5write(fname,'/px_dykes',px_dykes);
h5write(fname,'/py_dykes',py_dykes);
fname = [sim_dir '/markers.h5'];
h5create(fname, '/0/mx',size(mx),'ChunkSize',size(mx),'Deflate',5);
h5create(fname, '/0/my',size(my),'ChunkSize',size(my),'Deflate',5);
h5create(fname, '/0/mT',size(my),'ChunkSize',size(my),'Deflate',5);
h5write(fname,'/0/mx',mx);
h5write(fname,'/0/my',my);
h5write(fname,'/0/mT',mT);
% copy source code
 %{system(['copy ', sim_files, ' ', sim_dir,'\']);%}
% system(['copy ', sim_driver, '.m  ', sim_dir,'\']);
% % run CUDA
% cur_dir=pwd;
cd(sim_dir);
% system(['start ',sim_name,'.exe']);
% cd(cur_dir);
%{
system(['nvcc -ccbin ' cuda_ccbin                 ...
    '         -arch=' cuda_arch                   ...
    '         -Xcompiler "/MD"'                   ...
    '         -O3 -std=c++17 '                    ...
    '         -DGPU_ID='  num2str(gpuid)          ...
    '         -DNDIGITS=' num2str(ndigits)        ...
    '         -I ' '"' [hdf5_path '\include'] '"' ...
    '         -L ' '"' [hdf5_path '\lib'] '"'     ...
    '         -DH5_BUILT_AS_DYNAMIC_LIB'          ...
    '                   ' cu_name                 ...
    '         -lhdf5'                             ...
    ' -o ' exe_name]);
% cd ..
%}
