using CoordinateTransformations

function d2dm_melt_fraction(Temp)
	return inter(Temp)
end


	dykes_crystalinity = 1 .- Vector{Float64}([1, 1, 0.9904761904761905, 0.9439153277684773, 0.9280423118954614, 0.9121692960224456, 0.8772486611018104, 0.8306877983940973, 0.775661359514509, 0.7185185023716518, 0.6677248515780011, 0.6042327880859376, 0.5460317460317462, 0.4793650793650794, 0.41164017934647823, 0.35343908885168657, 0.2793650793650794, 0.2338624015687006, 0.19153432694692474, 0.13015873015873017, 0.09735446506076402, 0.08783065553695447, 0.08042321583581348, 0.06349206349206352, 0.04444444444444443, 0.022222222222222397, 0.009523809523809728, 0.008465576171874977,  0.007,  0.006,  0.005,  0.004,  0.003,  0.002, 0])
	dykes_temp = Vector{Float64}([0, 703.9651009116576, 739.2545589231133, 747.5812835431677, 765.0277420751164, 786.8358424662811, 802.2997596178105, 812.609025618284, 820.9357502383384, 823.3148144154968, 828.0729427698136, 830.8485297437112, 831.8485297437112, 835.2101353012887, 837.5891994784471, 838.3996673898679, 839.5717771605048, 843.1403734262424, 851.0705752495575, 859.0007770728726, 878.4298132868789, 907.7715927046195, 924.4250419447282, 953.370334867368, 987.8667654361645, 1018.3980769424843, 1044.1712963961259, 1067.9619381677096, 1092.9421120278728, 1120.697836560295, 1139.3338634824613, 1157.176844811149, 1170.658184280621, 1185.7256149370496, 2000])

	#d2dm_campi_rhyolite::Interpolations.MonotonicInterpolation = interpolate(dykes_temp, dykes_crystalinity, SteffenMonotonicInterpolation())
	#d2dm_campi_rhyolite::Interpolations.MonotonicInterpolation = interpolate(dykes_temp, dykes_crystalinity, SteffenMonotonicInterpolation())


	nodes = (dykes_temp,)
	itp = interpolate(nodes, dykes_crystalinity, Gridded(Linear()))
	cuitp = adapt(CuArray{eltype(dykes_temp)}, itp);

#	cuitp = adapt(CuDeviceArray{Float64}, d2dm_campi_rhyolite);
	#cuitp = adapt(CuArray{Float64}, d2dm_campi_rhyolite);
	#
#itp = (dykes_temp, dykes_crystalinity) # only BSpline is tested (Lanczos should also work)
#	itp = dykes_temp, interpolate(dykes_crystalinity, Gridded(Linear()))
#	cuitp = adapt(CuArray,itp)

function d2dm_test(T, a)
	ip = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	# b::Float64 = T[ip]
	# a[ip] = cuitp(b)
	T[ip] = cuitp(T)
	return
end

# function Adapt.adapt_structure(to, itp::Interpolations.MonotonicInterpolation{T,N,<:Any,IT,Axs}) where {T,N,IT,Axs}
#     coefs = Adapt.adapt_structure(to, itp.coefs)
#     Tcoefs = typeof(coefs)
#     Interpolations.MonotonicInterpolation{T,N,Tcoefs,IT,Axs}(coefs, itp.parentaxes, itp.it)
# end
#
# function Adapt.adapt_structure(to, itp::Interpolations.BSplineInterpolation{T,N,<:Any,IT,Axs}) where {T,N,IT,Axs}
# 		coefs = Adapt.adapt_structure(to, itp.coefs)
# 		Tcoefs = typeof(coefs)
# 		Interpolations.BSplineInterpolation{T,N,Tcoefs,IT,Axs}(coefs, itp.parentaxes, itp.it)
# 	end
#


function dykes_muskh()

	# Ts = CuArray(700.0:1.0:1100.0)
	# a = CuArray(700.0:1.0:1100.0)

	#	blockSize = (1, 1)
	#	gridSize = (1, 1)
	#	@cuda blocks = gridSize threads = blockSize d2dm_test(Ts, a)
	#	synchronize()

	#Ts_y = cuitp(Ts[1])
	#a= Array{Float64,1}(undef, length(Ts))

	# h_Ts = Array(700.0:1.0:1100.0)
	# h_a = Array(700.0:1.0:1100.0)
	#
	# Ts = cuitp.(Ts)
	#
	# copyto!(h_Ts, Ts)
	# copyto!(h_a, a)
	#
	# p = Plots.plot(h_Ts, h_h_a, xlims = [700, 1100] )

	# a = randn(Float64, 401)
	# itp = linear_interpolation(dykes_crystalinity, dykes_temp) # only BSpline is tested (Lanczos should also work)
	# cuitp = adapt(CuArray,itp) # you need to transfer the data into GPU's memory
	#
	# h_Ts = 800.0:1:900.0
	# h_a = cuitp.(800.0:1:900.0) # 20001×20001 CuArray{ComplexF32, 2}
	# h_h_a = Array{Float64, 1}(undef, length(h_Ts))
	#
	# println(typeof(h_Ts))
	# println(typeof(h_a))
	#
	# copyto!(h_h_a, h_a)

	#cuitp.(CUDA.randn(100),CUDA.randn(100)) # 100 CuArray{ComplexF32, 1}


	#p = Plots.plot(h_Ts, h_h_a, xlims = [700, 1100] )

	#p = Plots.plot(Ts, Ts_y, xlims = [700, 1100] )
	#p = Plots.plot!(dykes_temp, dykes_crystalinity, seriestype=:scatter)

	#
	# nodes = (dykes_temp,)
	# itp = interpolate(nodes, dykes_crystalinity, Gridded(Linear()))
	# sT_x = 700:1:1100
	# ans = [itp(x) for x in sT_x]
	#
	#
	#
	# p = Plots.plot(sT_x, ans, xlims = [700, 1100] )
	# p = Plots.plot!(dykes_temp, dykes_crystalinity, seriestype=:scatter)

# Data
#
# m=0.75; #Variable in Joukovskiy equasion
# r1=1;
# nu=0.3; #poussion coefficient
# eta=(1-2*nu)/(1-nu)/2;
# kappa = 3-4*nu;
# G = 1;
# Pcr = 1; # Fluid pressure on cavity
# Po  = 0; # Fluid pressure on external boundary
# rc = 20; # rho_*
# R=1;
# r2=2; # 1<=r2<=rc parameter which control the plotted area
# C=1;
#
# ro= collect(r1:(r2-r1)/150:r2);
# phi = collect(0:1/150:2*pi+pi/1e1);
# ro_cart = zeros(length(ro))
# phi_cart = zeros(length(phi))
#
#
# Rho = zeros(length(ro),length(phi));
# alpha = zeros(length(ro),length(phi));
# Srr = zeros(length(ro),length(phi));
#
#
#
# x = range(-2, 2, length=10)
# y = range(-2, 2, length=10)
#
# #println([x for _ in y for x in x])
#
#
# #println(ro_cart)
#
# #p = Plots.heatmap(ro, phi, Rho', title="Interpolated heatmap")
# #p = Plots.heatmap(ro_cart, phi_cart, Rho', title="Interpolated heatmap")
#
#  # x = ro
#  # y = phi
#  # z = Rho
#  #
# #p = Plots.scatter([x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data", clim=(-2,2))
#
#
# for i = 1:length(ro)
# 	for j = 1:length(phi)
# 		Rho[i,j]=ro[i];
# 		rho = ro[i];
# 		alpha[i,j]=phi[j];
# 		upsilon=phi[j];
# 		#Srr(i,j) =  (eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32))+eta*rho^2*Pcr*m^4*log(1/(rc^8))+eta*Pcr*m^4*log(rho^8*rc^8)+eta*rho^6*Pcr*log(rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m*rho^6*Po*cos(2*upsilon)*log(rho^32)+eta*m^2*rho^2*Pcr*log(1/(rc^8))+eta*rho^2*Po*m^4*log(rc^8)+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+8*eta*Po*m^4+24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr-8*eta*rho^6*Pcr*m^2+8*eta*rho^6*Po*m^2-8*eta*Pcr*m^4-24*eta*m^2*rho^2*Po+eta*rho^6*Pcr*m^2*log(rc^8)+8*eta*rho^2*Pcr*m^4-8*eta*rho^2*Po*m^4+8*eta*rho^2*Pcr*m^3*cos(2*upsilon)-8*eta*m^3*Pcr*cos(2*upsilon)+8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))+eta*rho^6*Po*log(1/(rc^8))+24*eta*m*rho^4*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)-24*eta*m*rho^6*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-8*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)-24*eta*m*rho^4*Pcr*cos(2*upsilon)+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32)+24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*m^2*rho^2*Po*log(rc^8)+eta*rho^6*Po*m^2*log(1/(rc^8))+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/(rho^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
#
#
# 		Srr[i,j] = ro[i]
#
# 		# Stt(i,j)= (eta*Pcr*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m^2*rho^2*Pcr*log(rc^8)+eta*rho^6*Po*m^2*log(rc^8)+eta*rho^2*Pcr*m^4*log(rc^8)+eta*m^2*rho^2*Po*log(1/(rc^8))+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+16*eta*Po*m^4-24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr+8*eta*rho^6*Pcr*m^2-8*eta*rho^6*Po*m^2-16*eta*Pcr*m^4+24*eta*m^2*rho^2*Po-8*eta*rho^2*Pcr*m^4+8*eta*rho^2*Po*m^4+56*eta*rho^2*Pcr*m^3*cos(2*upsilon)+8*eta*m^3*Pcr*cos(2*upsilon)-8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)+24*eta*m*rho^6*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-56*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)+24*eta*m*rho^4*Pcr*cos(2*upsilon)-24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*rho^6*Pcr*log(1/(rc^8))+eta*rho^6*Pcr*m^2*log(1/(rc^8))+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32*rc^32)+eta*rho^6*Po*log(rc^8)+eta*rho^2*Po*m^4*log(1/(rc^8))+eta*m*rho^6*Po*cos(2*upsilon)*log(1/rc^32*rho^32)+8*eta*rho^8*Pcr-8*eta*rho^8*Po+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/rho^32*rc^32)+eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
# 		# Srt(i,j) = eta*m*sin(2*upsilon)*(-2*rho^6*Pcr*log(rc)+2*rho^2*log(rc)*Po*m^2-2*rho^4*log(rc)*Po*m^2+2*rho^4*Pcr*log(rc)*m^2+2*m*Pcr*rho^2*cos(2*upsilon)-2*m*Po*rho^2*cos(2*upsilon)-2*rho^4*Pcr*m^2+2*rho^6*log(rc)*Po+2*rho^4*Po*m^2-2*m*rho^4*Pcr*cos(2*upsilon)-m^2*Pcr+m^2*Po-3*rho^2*Po*m^2+3*rho^4*Po-3*rho^4*Pcr+3*rho^2*Pcr*m^2-3*rho^6*Po+3*rho^6*Pcr+2*m*rho^4*Po*cos(2*upsilon)+2*rho^4*Pcr*log(rc)-2*rho^2*Pcr*log(rc)*m^2-2*rho^4*log(rc)*Po)/log(rc)/(4*m^2*rho^4*cos(2*upsilon)^2+2*m^2*rho^4-4*rho^6*m*cos(2*upsilon)-4*rho^2*m^3*cos(2*upsilon)+rho^8+m^4);
# 		#
# 		#
# 		# Ux(i,j)  = -1/8*eta*R*cos(upsilon)*(11*m*rho^4*Pcr-11*m*rho^4*Po+kappa*rho^6*Pcr+4*m^3*log(rho)*Po-4*m^3*log(rho)*Pcr+5*rho^2*Po*m^2-kappa*rho^6*Po+4*rho^2*Po*m^3-3*kappa*m^3*Pcr+3*kappa*m^3*Po-4*rho^2*Pcr*m^3+12*m*Po*rho^2-12*rho^2*m*Pcr-4*rho^2*m*log(rc)*Po+2*kappa*log(rc)*rho^6*Po+4*rho^2*Pcr*log(rc)*m^3+4*kappa*log(rc)*m^3*Pcr-20*rho^4*m*Pcr*cos(upsilon)^2+6*rho^4*m*log(rc)*Po+20*m*Po*rho^4*cos(upsilon)^2+12*Pcr*m^2*cos(upsilon)^2*rho^2-16*m*Po*cos(upsilon)^2*rho^2-12*Po*m^2*cos(upsilon)^2*rho^2+16*m*Pcr*cos(upsilon)^2*rho^2+Pcr*m^3+4*Po*m^2-rho^6*Po+rho^6*Pcr-5*rho^2*Pcr*m^2-8*kappa*log(rc)*m*Po*rho^4*cos(upsilon)^2+8*kappa*log(rc)*m^2*Pcr*rho^2+2*kappa*log(rc)*m*Po*rho^4-2*kappa*log(rc)*m^2*Po*rho^2-4*kappa*rho^4*Pcr*m*cos(upsilon)^2+4*kappa*log(rc)*m*Pcr*rho^4+4*kappa*m*Po*rho^4*cos(upsilon)^2-8*rho^4*m*log(rc)*Po*cos(upsilon)^2+16*cos(upsilon)^2*Pcr*rho^2*log(rho)*m^2+4*rho^4*Pcr*m^2-4*rho^4*Po*m^2-5*kappa*rho^2*Pcr*m^2+5*kappa*m^2*Po*rho^2+16*cos(upsilon)^2*m*log(rho)*Pcr*rho^4-16*cos(upsilon)^2*rho^2*Po*log(rho)*m^2-16*cos(upsilon)^2*m*log(rho)*Po*rho^4+8*kappa*log(rc)*m^2*Po*cos(upsilon)^2*rho^2-16*kappa*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2-12*kappa*m^2*Po*cos(upsilon)^2*rho^2-16*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2+4*rho^6*Pcr*log(rc)+4*rho^4*log(rc)*Po-4*rho^4*Pcr*log(rc)+8*log(rc)*m^2*Po*cos(upsilon)^2*rho^2+12*kappa*m^2*Pcr*cos(upsilon)^2*rho^2-2*rho^6*log(rc)*Po-4*Pcr*m^2-4*rho^4*Pcr*log(rc)*m^2+4*rho^4*log(rc)*Po*m^2+12*rho^2*Pcr*log(rc)*m^2-6*rho^2*log(rc)*Po*m^2-Po*m^3+kappa*m*Po*rho^4-kappa*rho^4*Pcr*m-12*Pcr*rho^2*log(rho)*m^2+12*rho^2*Po*log(rho)*m^2-4*rho^2*log(rc)*Po*m^3-12*m*log(rho)*Pcr*rho^4+12*m*log(rho)*Po*rho^4-2*kappa*log(rc)*m^3*Po+4*rho^2*m*Pcr*log(rc)+4*rho^6*Po*log(rho)-4*Pcr*rho^6*log(rho)+2*log(rc)*m^3*Po)/rho/log(rc)/G/(-m^2+4*m*rho^2*cos(upsilon)^2-2*m*rho^2-rho^4);
# 		#
# 		# Uy(i,j)  = -1/8*eta*R*sin(upsilon)*(-9*m*rho^4*Pcr+9*m*rho^4*Po-kappa*rho^6*Pcr+4*m^3*log(rho)*Po-4*m^3*log(rho)*Pcr+7*rho^2*Po*m^2+kappa*rho^6*Po+4*rho^2*Po*m^3-3*kappa*m^3*Pcr+3*kappa*m^3*Po-4*rho^2*Pcr*m^3-4*m*Po*rho^2+4*rho^2*m*Pcr-4*rho^2*m*log(rc)*Po-2*kappa*log(rc)*rho^6*Po+4*rho^2*Pcr*log(rc)*m^3+4*kappa*log(rc)*m^3*Pcr+20*rho^4*m*Pcr*cos(upsilon)^2-2*rho^4*m*log(rc)*Po-20*m*Po*rho^4*cos(upsilon)^2+12*Pcr*m^2*cos(upsilon)^2*rho^2+16*m*Po*cos(upsilon)^2*rho^2-12*Po*m^2*cos(upsilon)^2*rho^2-16*m*Pcr*cos(upsilon)^2*rho^2+Pcr*m^3-4*Po*m^2+rho^6*Po-rho^6*Pcr-7*rho^2*Pcr*m^2+8*kappa*log(rc)*m*Po*rho^4*cos(upsilon)^2+8*kappa*log(rc)*m^2*Pcr*rho^2-6*kappa*log(rc)*m*Po*rho^4-6*kappa*log(rc)*m^2*Po*rho^2+4*kappa*rho^4*Pcr*m*cos(upsilon)^2+4*kappa*log(rc)*m*Pcr*rho^4-4*kappa*m*Po*rho^4*cos(upsilon)^2+8*rho^4*m*log(rc)*Po*cos(upsilon)^2+16*cos(upsilon)^2*Pcr*rho^2*log(rho)*m^2-4*rho^4*Pcr*m^2+4*rho^4*Po*m^2-7*kappa*rho^2*Pcr*m^2+7*kappa*m^2*Po*rho^2-16*cos(upsilon)^2*m*log(rho)*Pcr*rho^4-16*cos(upsilon)^2*rho^2*Po*log(rho)*m^2+16*cos(upsilon)^2*m*log(rho)*Po*rho^4+8*kappa*log(rc)*m^2*Po*cos(upsilon)^2*rho^2-16*kappa*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2-12*kappa*m^2*Po*cos(upsilon)^2*rho^2-16*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2-4*rho^6*Pcr*log(rc)-4*rho^4*log(rc)*Po+4*rho^4*Pcr*log(rc)+8*log(rc)*m^2*Po*cos(upsilon)^2*rho^2+12*kappa*m^2*Pcr*cos(upsilon)^2*rho^2+2*rho^6*log(rc)*Po+4*Pcr*m^2+4*rho^4*Pcr*log(rc)*m^2-4*rho^4*log(rc)*Po*m^2+4*rho^2*Pcr*log(rc)*m^2-2*rho^2*log(rc)*Po*m^2-Po*m^3+5*kappa*m*Po*rho^4-5*kappa*rho^4*Pcr*m-4*Pcr*rho^2*log(rho)*m^2+4*rho^2*Po*log(rho)*m^2-4*rho^2*log(rc)*Po*m^3+4*m*log(rho)*Pcr*rho^4-4*m*log(rho)*Po*rho^4-2*kappa*log(rc)*m^3*Po+4*rho^2*m*Pcr*log(rc)-4*rho^6*Po*log(rho)+4*Pcr*rho^6*log(rho)+2*log(rc)*m^3*Po)/rho/log(rc)/G/(m^2-4*m*rho^2*cos(upsilon)^2+2*m*rho^2+rho^4);
# 		#
# 		# Ur(i,j)  =  1/8*R*eta*(-4*rho^2*Pcr*log(rc)*m^2+4*m^2*log(rho)*Pcr-4*m^2*log(rho)*Po-4*rho^2*log(rc)*m*Po*cos(2*upsilon)-2*rho^4*log(rc)*Po+4*rho^4*Pcr*log(rc)+4*rho^2*log(rc)*Po*m^2+4*kappa*log(rc)*m*Pcr*rho^2*cos(2*upsilon)+4*rho^2*log(rc)*m*Pcr*cos(2*upsilon)+2*kappa*log(rc)*m^2*Po-4*kappa*log(rc)*m^2*Pcr+2*kappa*log(rc)*rho^4*Po-4*m*Po*cos(2*upsilon)-4*kappa*rho^2*Pcr*m*cos(2*upsilon)+4*kappa*rho^2*Po*m*cos(2*upsilon)-2*log(rc)*m^2*Po-3*kappa*Po*m^2+kappa*rho^4*Pcr+4*rho^2*log(rc)*Po+3*kappa*Pcr*m^2-4*Pcr*rho^2*log(rc)-kappa*rho^4*Po+4*m*Pcr*cos(2*upsilon)+Po*m^2-Pcr*m^2-8*m*Pcr*rho^2*cos(2*upsilon)+8*m*Po*rho^2*cos(2*upsilon)-4*rho^4*Pcr*log(rho)+4*rho^4*Po*log(rho)-4*kappa*log(rc)*rho^2*Po*m*cos(2*upsilon)-rho^4*Po+rho^4*Pcr-4*rho^2*Po*m^2+4*rho^2*Pcr*m^2)/(-2*m*rho^2*cos(2*upsilon)+rho^4+m^2)^(1/2)/rho/G/log(rc);
# 		# Ut(i,j)  = -1/4*R*eta*m*sin(2*upsilon)*(2*Pcr*rho^2*log(rc)-kappa*rho^2*Pcr+rho^2*kappa*Po+2*kappa*log(rc)*Pcr*rho^2-rho^2*Po+rho^2*Pcr-2*Pcr+2*Po-4*rho^2*Pcr*log(rho)+4*rho^2*Po*log(rho))/(-2*m*rho^2*cos(2*upsilon)+rho^4+m^2)^(1/2)/rho/G/log(rc);
# 	end
# end
#
# 	Z=Rho.*exp.(im*alpha);
# 	X=real(1/2*(Z+m./Z));
# 	Y=imag(1/2*(Z+m./Z));
# 	#println(X)
#
#
# 	x = range(-2, 2, length=10)
# 	y = range(-2, 2, length=10)
#
# 	itp = interpolate(X,Y, Srr, Gridded(Linear()))
#
# 	#p = Plots.heatmap(X, Y, Srr, title="Interpolated heatmap")
# 	p = Plots.scatter(X, Y, Srr, title="Interpolated heatmap")
#
# return p
#
	# x = range(-2, 2, length=10)
	# y = range(-2, 2, length=10)
	# z = @. cos(x) + sin(y')
	#
	# # Interpolation object (caches coefficients and such)
	# itp = LinearInterpolation((x, y), z)
	# # Fine grid
	# x2 = range(extrema(x)..., length=300)
	# y2 = range(extrema(y)..., length=200)
	# # Interpolate
	# z2 = [itp(x,y) for y in y2, x in x2]
	# # Plot
	# p = Plots.heatmap(x2, y2, z2, clim=(-2,2), title="Interpolated heatmap")
	# Plots.scatter!(p, [x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data", clim=(-2,2))
	

	# x = range(-3, 3, length=100)
	# y = range(-3, 3, length=100)
	#
	#
	# x2 = range(extrema(x)..., length=300)
	# y2 = range(extrema(y)..., length=200)
	#
	# z2 = [calc_Srr(x, y, m, Pcr, eta, rc, Po) for y in y2, x in x2]
	#
	# p = Plots.heatmap(x2, y2, z2, clim=(-3,3), title="Interpolated heatmap")
	
	#dykes_crystalinity = 1 .- Vector{Float64}([1, 1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0, 0])
	#dykes_temp = Vector{Float64}([0, 699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385, 1200])
	#dykes_temp = Vector{Float64}( [699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 118.2385]);
	#dykes_crystalinity = 1 .- Vector{Float64}([1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0]);


	# itp = itp = interpolate(A, BSpline(Linear()))
	#A_x = range(699.2355, 1186.2385, 51);
	#itp = DataInterpolations.LinearInterpolation(dykes_crystalinity, dykes_temp,extrapolate = true)


	# for i in eachindex(x)
	# 	y[i] = ForwardDiff.derivative(itp,y[i])
	# end
	
	#dykes_crystalinity = 1 .- Vector{Float64}([1, 1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0, 0])
	#dykes_temp = Vector{Float64}([0, 699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385, 1200])
	

	dykes_crystalinity = 1 .- Vector{Float64}([ 1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0 ])
	dykes_temp = Vector{Float64}([ 699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385])

	x = range(700, 1200, 600);


	#A_x = 0:1/52.0:1
	#A_x = 1.:52.0:52

	A_x = range(699.2355, 1186.2385, 51);
	itp = interpolate(dykes_crystalinity, BSpline(Cubic(Line(OnGrid()))))
	itp = Interpolations.scale(itp, A_x)
	itp = extrapolate(itp, Flat())

	y = itp(x)
	p = Plots.plot(x,y)
	
	cuitp = adapt(CuArray{eltype(dykes_temp)}, itp);

	y = only.(Interpolations.gradient.(Ref(cuitp), x))

	dmf_cuitp = adapt(CuArray{eltype(dykes_temp)}, itp);
	# p = Plots.plot( x, y)
	# for i in eachindex(x)
	# 	y[i] = ForwardDiff.derivative(itp,y[i])
	# end
	println(typeof(y))
	@CUDA.allowscalar Plots.plot!(p, x, y)
	
	Plots.plot!(p, dykes_temp, dykes_crystalinity, seriestype=:scatter)

	return p

end

function calc_Srr(point_x, point_y, m, Pcr, eta, rc, Po)
	upsilon, rho = cart2pol(point_x, point_y);
	Z_real = rho*exp(1i*upsilon);
	if(point_x >= 0)
		Zr_rev_z = Z_real + sqrt(Z_real^2 -m);
	else
		Zr_rev_z = Z_real - sqrt(Z_real^2 -m);
	end

	X=real(Zr_rev_z);
	Y=imag(Zr_rev_z);

	upsilon, rho = cart2pol(X, Y);
	if(rho>=1)
		Srr =  (eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32))+eta*rho^2*Pcr*m^4*log(1/(rc^8))+eta*Pcr*m^4*log(rho^8*rc^8)+eta*rho^6*Pcr*log(rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m*rho^6*Po*cos(2*upsilon)*log(rho^32)+eta*m^2*rho^2*Pcr*log(1/(rc^8))+eta*rho^2*Po*m^4*log(rc^8)+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+8*eta*Po*m^4+24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr-8*eta*rho^6*Pcr*m^2+8*eta*rho^6*Po*m^2-8*eta*Pcr*m^4-24*eta*m^2*rho^2*Po+eta*rho^6*Pcr*m^2*log(rc^8)+8*eta*rho^2*Pcr*m^4-8*eta*rho^2*Po*m^4+8*eta*rho^2*Pcr*m^3*cos(2*upsilon)-8*eta*m^3*Pcr*cos(2*upsilon)+8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))+eta*rho^6*Po*log(1/(rc^8))+24*eta*m*rho^4*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)-24*eta*m*rho^6*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-8*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)-24*eta*m*rho^4*Pcr*cos(2*upsilon)+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32)+24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*m^2*rho^2*Po*log(rc^8)+eta*rho^6*Po*m^2*log(1/(rc^8))+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/(rho^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
	else
		Srr = -1;
	end

	return Srr
end
