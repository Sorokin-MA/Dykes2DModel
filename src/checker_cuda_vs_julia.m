filename_julia = "julia_out.h5"
filename_cuda = "cuda_out.h5"

fd = H5F.open(filename_julia,'H5F_ACC_RDONLY','H5P_DEFAULT') ;
fd_info = H5F.get_info(fd);

T_julia = h5read(filename_julia,'/pT');
C_julia= h5read(filename_julia,'/C');

%T_julia = reshape(T_julia, 4000,[]);
C_julia = reshape(C_julia, 4000,[]);

fd = H5F.open(filename_cuda,'H5F_ACC_RDONLY','H5P_DEFAULT') ;
fd_info = H5F.get_info(fd);

T_cuda = h5read(filename_cuda,'/pT');
C_cuda= h5read(filename_cuda,'/C');

%T_cuda = reshape(T_cuda, 4000,[]);
C_cuda = reshape(C_cuda, 4000,[]);

max(T_julia-T_cuda(:))


%{
figure

%T_julia
subplot(3,2,1);
imshow(T_julia)
pcolor(T_julia);
title('T, julia')
shading interp;
colorbar

%C_julia
subplot(3,2,2);
imshow(C_julia)
pcolor(C_julia);
title('C, julia')
shading interp;
colorbar  

%T_cuda
subplot(3,2,3);
imshow(T_julia)
pcolor(T_julia);
title('T, cuda')
shading interp;
colorbar

%C_cuda
subplot(3,2,4);
imshow(C_julia)
pcolor(C_julia);
title('C, cuda')
shading interp;
colorbar  

%T_cuda
subplot(3,2,5);
imshow(T_julia-T_cuda)
pcolor(T_julia);
title('T, julia - T, cuda')
shading interp;
colorbar

%C_cuda
subplot(3,2,6);
imshow(C_julia-C_cuda)
pcolor(C_julia);
title('C, julia - C, cuda')
shading interp;
colorbar  

%T_julia = h5read(filename_1_pr,'/T');
%C_julia= h5read(filename_1_pr,'/C');
%}