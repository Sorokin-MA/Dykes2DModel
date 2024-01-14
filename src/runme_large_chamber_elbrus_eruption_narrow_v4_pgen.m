clear;
seed = 123;
rng(seed);
% program config
% cur_dir    =pwd;

sim_dir    = 'init_data';												%directory name where will be compilled files
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
delete([sim_dir '/*']);

% physics
% dimensionally independent
Lx          = 20000; % x size of area, m
Ly          = 25000; % y siez of area, m
Lx_Ly       = Lx/Ly;
narrow_fact = 0.6;
dike_x_W    = 5000; %m 
dike_x_Wn    = dike_x_W*narrow_fact; %m 
dike_a_rng  = [100 1500]; %m
dike_b_rng  = [10 20]; %m

dike_x_rng  = [ (Lx-dike_x_W)/2 (Lx+dike_x_W)/2] ;
dike_x_rng_n  = [ (Lx-dike_x_Wn)/2 (Lx+dike_x_Wn)/2] ;

dike_y_rng  = [1000 22000];				%dikes y distribution
dike_t_rng  = [0.95*pi/2 1.05*pi/2];	%dykes time distribution
dike_to_sill = 21000;					%boundary where dykes turn yourself to sill, m
dz           = 5000;					%z dimension? i guess, m

Lam_r       = 1.5;			%thermal conductivity of rock, W/m/K
Lam_m       = 1.2;			%thermal conductivity of magma, W/m/K
rho         = 2650;			%density, kg/m^3
Cp          = 1350;			%scpecifiv heat capacity, J/kg/K
Lheat       = 3.5e5;		%Latent heat of melting, J/kg
T_top       = 100;			%temperature at depth 5 km, C
dTdy        = 20;			%how fast temperature decreasing with depth, K/km
T_magma     = 950;			%magma intrusion temperature, C
T_ch        = 700;			%?
Qv          = 0.038;		%m^3/s
dt          = 50*tyear;		%time
tfin        = 600e3*tyear;
terupt      = 600e3*tyear;


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

% distr       = random('exponential',alpha,1,Nsample);
distr = -alpha .* log(rand(1,Nsample, 'like', alpha)); % == expinv(u, mu)
rn          = (distr-min(distr))/(max(distr)-min(distr));
critVol     = 10.^(9+3*rn)/dz/(1-gamma);
% dike_x_rng  = [0.32 0.67]*Lx;

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
nout        = 10000;
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
indx=find(xs > dike_x_rng(1) & xs < dike_x_rng(2));
indy=find(ys >dike_y_rng(1)  &  ys <dike_y_rng(2));
T(indx,indy)=T_ch;
pcolor(x,y,T),shading flat,axis image;c= colorbar,drawnow

C           = zeros(nx,ny);
% generate dikes
Q      = 0;
dike_a = [];
dike_b = [];
dike_x = [];
dike_y = [];
dike_t = [];
dike_v = [];
Vtot=q*nt_erupt*dt;
Q_tsh       = 0.5*Vtot;
while Q < Vtot
    dike_a = [dike_a dike_a_rng(1) + diff(dike_a_rng)*rand];
    dike_b = [dike_b dike_b_rng(1) + diff(dike_b_rng)*rand];
    if Q < Q_tsh
        dike_x = [dike_x dike_x_rng(1) + diff(dike_x_rng)*rand];
    else
        dike_x = [dike_x dike_x_rng_n(1) + diff(dike_x_rng_n)*rand];
    end
    dike_y = [dike_y dike_y_rng(1) + diff(dike_y_rng)*rand];
    dike_t = [dike_t dike_t_rng(1) + diff(dike_t_rng)*rand];
    dike_v = [dike_v pi*dike_a(end)*dike_b(end)];
    Q      = Q + dike_v(end);
end
dike_v = [0 cumsum(dike_v)];
dv     = dike_v(end)/nt_erupt;
ndikes = diff(floor(interp1(dike_v,1:numel(dike_v),0:dv:dike_v(end))));
assert(numel(ndikes) == nt_erupt);
ndikes(numel(ndikes)+1:nt) = 0;
assert(numel(ndikes) == nt);
dike_npartcl = zeros(1,sum(ndikes));
dike_nmarker = zeros(1,sum(ndikes));
px_dike      = cell(sum(ndikes),1);
py_dike      = cell(sum(ndikes),1);
mx_dike      = cell(sum(ndikes),1);
my_dike      = cell(sum(ndikes),1);
dike_t(dike_y>=dike_to_sill) = dike_t(dike_y>=dike_to_sill) + pi/2;
for idike = 1:sum(ndikes)
    a              = dike_a(idike);
    b              = dike_b(idike);
    dikex0         = dike_x(idike);
    dikey0         = dike_y(idike);
    st             = sin(dike_t(idike));
    ct             = cos(dike_t(idike));
    % markers
    dikexs         = linspace(-a,a,round(2*a/dr));
    dikeys         = linspace(-b,b,round(2*b/dr));
    if isempty(dikexs);dikexs = 0;end
    if isempty(dikeys);dikeys = 0;end
    [dikex,dikey]  = ndgrid(dikexs,dikeys);
    dikex          = dikex(:);
    dikey          = dikey(:);
    outside        = (dikex.^2/a^2 + dikey.^2/b^2) > 1+eps;
    dikex(outside) = [];
    dikey(outside) = [];
    px_dike{idike} = dikex0 + dikex*ct - dikey*st;
    py_dike{idike} = dikey0 + dikex*st + dikey*ct;
    dike_npartcl(idike) = numel(px_dike{idike});
    % markers
    dikemxs         = linspace(-a,a,round(2*a/dmr));
    dikemys         = linspace(-b,b,round(2*b/dmr));
    if numel(dikemxs)<=1;dikemxs = 0;end
    if numel(dikemys)<=1;dikemys = 0;end
    [dikemx,dikemy]  = ndgrid(dikemxs,dikemys);
    dikemx          = dikemx(:);
    dikemy          = dikemy(:);
    outside         = (dikemx.^2/a^2 + dikemy.^2/b^2) > 1+eps;
    dikemx(outside) = [];
    dikemy(outside) = [];
    mx_dike{idike}  = dikex0 + dikemx*ct - dikemy*st;
    my_dike{idike}  = dikey0 + dikemx*st + dikemy*ct;
    dike_nmarker(idike) = numel(mx_dike{idike});
end
px_dikes     = cell2mat(px_dike);
py_dikes     = cell2mat(py_dike);
mx           = [mx;cell2mat(mx_dike)];
my           = [my;cell2mat(my_dike)];
mT           = T_top + dTdy/1e3*(Ly-my);
mT(mx > dike_x_rng(1) & mx < dike_x_rng(2) & my > dike_y_rng(1)  & my <dike_y_rng(2))=T_ch;


partcl_edges = [0 cumsum(dike_npartcl)];
marker_edges = [0 cumsum(dike_nmarker)];

% save data
fid        = fopen([sim_dir '/pa.bin'],'w');
fwrite(fid,[Lx Ly lam_r_rhoCp lam_m_rhoCp L_Cp T_top T_bot T_magma tsh gamma Ly_eruption nu G dt_diff dx dy eiter pic_amount],'double');
fwrite(fid,[pmlt nx ny nl nt niter nout nsub nerupt npartcl nmarker Nsample],'int32');
fwrite(fid,critVol,'double');
fwrite(fid,ndikes,'int32');
fwrite(fid,partcl_edges,'int32');
fwrite(fid,marker_edges,'int32');
fclose(fid);
fid        = fopen([sim_dir '/dikes.bin'],'w');
fwrite(fid,[dike_a dike_b dike_x dike_y dike_t],'double');
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
h5create(fname, '/px_dikes',size(px_dikes),'ChunkSize',size(px_dikes),'Deflate',5);
h5create(fname, '/py_dikes',size(py_dikes),'ChunkSize',size(py_dikes),'Deflate',5);
h5write(fname,'/px_dikes',px_dikes);
h5write(fname,'/py_dikes',py_dikes);
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
