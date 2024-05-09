sim_files   = 'magma_chamber_eruption_rh_rh_particles_generation.*';
sim_name   = 'magma_chamber_eruption_rh_rh_particles_generation';
sim_dir    = '.';
cuda_ccbin = '"c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.34.31933\bin\Hostx64\x64"';
cuda_arch  = ['sm_61'];
gpuid      = 1;
nt          = 120000;

ndigits     = floor(log10(nt))+1;
hdf5_path  = 'C:\Program Files\HDF_Group\HDF5\1.14.3';
cu_name    = [sim_name '.cu'];
exe_name   = [sim_name '.exe'];

system(['copy ', sim_files, ' ', sim_dir,'\']);

cd(sim_dir);

system(['nvcc -g -G -ccbin ' cuda_ccbin                 ...
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