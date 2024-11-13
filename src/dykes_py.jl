using PyCall
@pyimport scipy.interpolate as si
#using ScatteredInterpolation 

function dykes_py()

	#Data
	m=0.75; #Variable in Joukovskiy equasion
	r1=1;
	nu=0.3; #poussion coefficient
	eta=(1-2*nu)/(1-nu)/2;
	kappa = 3-4*nu;
	G = 1;
	Pcr = 1; # Fluid pressure on cavity
	Po  = 0; # Fluid pressure on external boundary
	rc = 20; # rho_*
	R=1;
	r2=2; # 1<=r2<=rc parameter which control the plotted area
	C=1;

	ro= collect(r1:(r2-r1)/50:r2);
	phi = collect(0:1/50:2*pi+pi/1e1);
	ro_cart = zeros(length(ro))
	phi_cart = zeros(length(phi))


	Rho = zeros(length(ro),length(phi));
	alpha = zeros(length(ro),length(phi));
	Srr = zeros(length(ro),length(phi));


	x = range(-2, 2, length=10)
	y = range(-2, 2, length=10)

	#println([x for _ in y for x in x])


	#println(ro_cart)

	#p = Plots.heatmap(ro, phi, Rho', title="Interpolated heatmap")
	#p = Plots.heatmap(ro_cart, phi_cart, Rho', title="Interpolated heatmap")

	 # x = ro
	 # y = phi
	 # z = Rho
	 #
	#p = Plots.scatter([x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data", clim=(-2,2))


	for i = 1:length(ro)
		for j = 1:length(phi)
			Rho[i,j]=ro[i];
			rho = ro[i];
			alpha[i,j]=phi[j];
			upsilon=phi[j];
			Srr[i,j] =  (eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32))+eta*rho^2*Pcr*m^4*log(1/(rc^8))+eta*Pcr*m^4*log(rho^8*rc^8)+eta*rho^6*Pcr*log(rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m*rho^6*Po*cos(2*upsilon)*log(rho^32)+eta*m^2*rho^2*Pcr*log(1/(rc^8))+eta*rho^2*Po*m^4*log(rc^8)+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+8*eta*Po*m^4+24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr-8*eta*rho^6*Pcr*m^2+8*eta*rho^6*Po*m^2-8*eta*Pcr*m^4-24*eta*m^2*rho^2*Po+eta*rho^6*Pcr*m^2*log(rc^8)+8*eta*rho^2*Pcr*m^4-8*eta*rho^2*Po*m^4+8*eta*rho^2*Pcr*m^3*cos(2*upsilon)-8*eta*m^3*Pcr*cos(2*upsilon)+8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))+eta*rho^6*Po*log(1/(rc^8))+24*eta*m*rho^4*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)-24*eta*m*rho^6*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-8*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)-24*eta*m*rho^4*Pcr*cos(2*upsilon)+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32)+24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*m^2*rho^2*Po*log(rc^8)+eta*rho^6*Po*m^2*log(1/(rc^8))+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/(rho^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));


			# Srr[i,j] = ro[i]

			# Stt(i,j)= (eta*Pcr*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m^2*rho^2*Pcr*log(rc^8)+eta*rho^6*Po*m^2*log(rc^8)+eta*rho^2*Pcr*m^4*log(rc^8)+eta*m^2*rho^2*Po*log(1/(rc^8))+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+16*eta*Po*m^4-24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr+8*eta*rho^6*Pcr*m^2-8*eta*rho^6*Po*m^2-16*eta*Pcr*m^4+24*eta*m^2*rho^2*Po-8*eta*rho^2*Pcr*m^4+8*eta*rho^2*Po*m^4+56*eta*rho^2*Pcr*m^3*cos(2*upsilon)+8*eta*m^3*Pcr*cos(2*upsilon)-8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)+24*eta*m*rho^6*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-56*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)+24*eta*m*rho^4*Pcr*cos(2*upsilon)-24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*rho^6*Pcr*log(1/(rc^8))+eta*rho^6*Pcr*m^2*log(1/(rc^8))+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32*rc^32)+eta*rho^6*Po*log(rc^8)+eta*rho^2*Po*m^4*log(1/(rc^8))+eta*m*rho^6*Po*cos(2*upsilon)*log(1/rc^32*rho^32)+8*eta*rho^8*Pcr-8*eta*rho^8*Po+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/rho^32*rc^32)+eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
			# Srt(i,j) = eta*m*sin(2*upsilon)*(-2*rho^6*Pcr*log(rc)+2*rho^2*log(rc)*Po*m^2-2*rho^4*log(rc)*Po*m^2+2*rho^4*Pcr*log(rc)*m^2+2*m*Pcr*rho^2*cos(2*upsilon)-2*m*Po*rho^2*cos(2*upsilon)-2*rho^4*Pcr*m^2+2*rho^6*log(rc)*Po+2*rho^4*Po*m^2-2*m*rho^4*Pcr*cos(2*upsilon)-m^2*Pcr+m^2*Po-3*rho^2*Po*m^2+3*rho^4*Po-3*rho^4*Pcr+3*rho^2*Pcr*m^2-3*rho^6*Po+3*rho^6*Pcr+2*m*rho^4*Po*cos(2*upsilon)+2*rho^4*Pcr*log(rc)-2*rho^2*Pcr*log(rc)*m^2-2*rho^4*log(rc)*Po)/log(rc)/(4*m^2*rho^4*cos(2*upsilon)^2+2*m^2*rho^4-4*rho^6*m*cos(2*upsilon)-4*rho^2*m^3*cos(2*upsilon)+rho^8+m^4);
			#
			#
			# Ux(i,j)  = -1/8*eta*R*cos(upsilon)*(11*m*rho^4*Pcr-11*m*rho^4*Po+kappa*rho^6*Pcr+4*m^3*log(rho)*Po-4*m^3*log(rho)*Pcr+5*rho^2*Po*m^2-kappa*rho^6*Po+4*rho^2*Po*m^3-3*kappa*m^3*Pcr+3*kappa*m^3*Po-4*rho^2*Pcr*m^3+12*m*Po*rho^2-12*rho^2*m*Pcr-4*rho^2*m*log(rc)*Po+2*kappa*log(rc)*rho^6*Po+4*rho^2*Pcr*log(rc)*m^3+4*kappa*log(rc)*m^3*Pcr-20*rho^4*m*Pcr*cos(upsilon)^2+6*rho^4*m*log(rc)*Po+20*m*Po*rho^4*cos(upsilon)^2+12*Pcr*m^2*cos(upsilon)^2*rho^2-16*m*Po*cos(upsilon)^2*rho^2-12*Po*m^2*cos(upsilon)^2*rho^2+16*m*Pcr*cos(upsilon)^2*rho^2+Pcr*m^3+4*Po*m^2-rho^6*Po+rho^6*Pcr-5*rho^2*Pcr*m^2-8*kappa*log(rc)*m*Po*rho^4*cos(upsilon)^2+8*kappa*log(rc)*m^2*Pcr*rho^2+2*kappa*log(rc)*m*Po*rho^4-2*kappa*log(rc)*m^2*Po*rho^2-4*kappa*rho^4*Pcr*m*cos(upsilon)^2+4*kappa*log(rc)*m*Pcr*rho^4+4*kappa*m*Po*rho^4*cos(upsilon)^2-8*rho^4*m*log(rc)*Po*cos(upsilon)^2+16*cos(upsilon)^2*Pcr*rho^2*log(rho)*m^2+4*rho^4*Pcr*m^2-4*rho^4*Po*m^2-5*kappa*rho^2*Pcr*m^2+5*kappa*m^2*Po*rho^2+16*cos(upsilon)^2*m*log(rho)*Pcr*rho^4-16*cos(upsilon)^2*rho^2*Po*log(rho)*m^2-16*cos(upsilon)^2*m*log(rho)*Po*rho^4+8*kappa*log(rc)*m^2*Po*cos(upsilon)^2*rho^2-16*kappa*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2-12*kappa*m^2*Po*cos(upsilon)^2*rho^2-16*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2+4*rho^6*Pcr*log(rc)+4*rho^4*log(rc)*Po-4*rho^4*Pcr*log(rc)+8*log(rc)*m^2*Po*cos(upsilon)^2*rho^2+12*kappa*m^2*Pcr*cos(upsilon)^2*rho^2-2*rho^6*log(rc)*Po-4*Pcr*m^2-4*rho^4*Pcr*log(rc)*m^2+4*rho^4*log(rc)*Po*m^2+12*rho^2*Pcr*log(rc)*m^2-6*rho^2*log(rc)*Po*m^2-Po*m^3+kappa*m*Po*rho^4-kappa*rho^4*Pcr*m-12*Pcr*rho^2*log(rho)*m^2+12*rho^2*Po*log(rho)*m^2-4*rho^2*log(rc)*Po*m^3-12*m*log(rho)*Pcr*rho^4+12*m*log(rho)*Po*rho^4-2*kappa*log(rc)*m^3*Po+4*rho^2*m*Pcr*log(rc)+4*rho^6*Po*log(rho)-4*Pcr*rho^6*log(rho)+2*log(rc)*m^3*Po)/rho/log(rc)/G/(-m^2+4*m*rho^2*cos(upsilon)^2-2*m*rho^2-rho^4);
			#
			# Uy(i,j)  = -1/8*eta*R*sin(upsilon)*(-9*m*rho^4*Pcr+9*m*rho^4*Po-kappa*rho^6*Pcr+4*m^3*log(rho)*Po-4*m^3*log(rho)*Pcr+7*rho^2*Po*m^2+kappa*rho^6*Po+4*rho^2*Po*m^3-3*kappa*m^3*Pcr+3*kappa*m^3*Po-4*rho^2*Pcr*m^3-4*m*Po*rho^2+4*rho^2*m*Pcr-4*rho^2*m*log(rc)*Po-2*kappa*log(rc)*rho^6*Po+4*rho^2*Pcr*log(rc)*m^3+4*kappa*log(rc)*m^3*Pcr+20*rho^4*m*Pcr*cos(upsilon)^2-2*rho^4*m*log(rc)*Po-20*m*Po*rho^4*cos(upsilon)^2+12*Pcr*m^2*cos(upsilon)^2*rho^2+16*m*Po*cos(upsilon)^2*rho^2-12*Po*m^2*cos(upsilon)^2*rho^2-16*m*Pcr*cos(upsilon)^2*rho^2+Pcr*m^3-4*Po*m^2+rho^6*Po-rho^6*Pcr-7*rho^2*Pcr*m^2+8*kappa*log(rc)*m*Po*rho^4*cos(upsilon)^2+8*kappa*log(rc)*m^2*Pcr*rho^2-6*kappa*log(rc)*m*Po*rho^4-6*kappa*log(rc)*m^2*Po*rho^2+4*kappa*rho^4*Pcr*m*cos(upsilon)^2+4*kappa*log(rc)*m*Pcr*rho^4-4*kappa*m*Po*rho^4*cos(upsilon)^2+8*rho^4*m*log(rc)*Po*cos(upsilon)^2+16*cos(upsilon)^2*Pcr*rho^2*log(rho)*m^2-4*rho^4*Pcr*m^2+4*rho^4*Po*m^2-7*kappa*rho^2*Pcr*m^2+7*kappa*m^2*Po*rho^2-16*cos(upsilon)^2*m*log(rho)*Pcr*rho^4-16*cos(upsilon)^2*rho^2*Po*log(rho)*m^2+16*cos(upsilon)^2*m*log(rho)*Po*rho^4+8*kappa*log(rc)*m^2*Po*cos(upsilon)^2*rho^2-16*kappa*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2-12*kappa*m^2*Po*cos(upsilon)^2*rho^2-16*log(rc)*m^2*Pcr*cos(upsilon)^2*rho^2-4*rho^6*Pcr*log(rc)-4*rho^4*log(rc)*Po+4*rho^4*Pcr*log(rc)+8*log(rc)*m^2*Po*cos(upsilon)^2*rho^2+12*kappa*m^2*Pcr*cos(upsilon)^2*rho^2+2*rho^6*log(rc)*Po+4*Pcr*m^2+4*rho^4*Pcr*log(rc)*m^2-4*rho^4*log(rc)*Po*m^2+4*rho^2*Pcr*log(rc)*m^2-2*rho^2*log(rc)*Po*m^2-Po*m^3+5*kappa*m*Po*rho^4-5*kappa*rho^4*Pcr*m-4*Pcr*rho^2*log(rho)*m^2+4*rho^2*Po*log(rho)*m^2-4*rho^2*log(rc)*Po*m^3+4*m*log(rho)*Pcr*rho^4-4*m*log(rho)*Po*rho^4-2*kappa*log(rc)*m^3*Po+4*rho^2*m*Pcr*log(rc)-4*rho^6*Po*log(rho)+4*Pcr*rho^6*log(rho)+2*log(rc)*m^3*Po)/rho/log(rc)/G/(m^2-4*m*rho^2*cos(upsilon)^2+2*m*rho^2+rho^4);
			#
			# Ur(i,j)  =  1/8*R*eta*(-4*rho^2*Pcr*log(rc)*m^2+4*m^2*log(rho)*Pcr-4*m^2*log(rho)*Po-4*rho^2*log(rc)*m*Po*cos(2*upsilon)-2*rho^4*log(rc)*Po+4*rho^4*Pcr*log(rc)+4*rho^2*log(rc)*Po*m^2+4*kappa*log(rc)*m*Pcr*rho^2*cos(2*upsilon)+4*rho^2*log(rc)*m*Pcr*cos(2*upsilon)+2*kappa*log(rc)*m^2*Po-4*kappa*log(rc)*m^2*Pcr+2*kappa*log(rc)*rho^4*Po-4*m*Po*cos(2*upsilon)-4*kappa*rho^2*Pcr*m*cos(2*upsilon)+4*kappa*rho^2*Po*m*cos(2*upsilon)-2*log(rc)*m^2*Po-3*kappa*Po*m^2+kappa*rho^4*Pcr+4*rho^2*log(rc)*Po+3*kappa*Pcr*m^2-4*Pcr*rho^2*log(rc)-kappa*rho^4*Po+4*m*Pcr*cos(2*upsilon)+Po*m^2-Pcr*m^2-8*m*Pcr*rho^2*cos(2*upsilon)+8*m*Po*rho^2*cos(2*upsilon)-4*rho^4*Pcr*log(rho)+4*rho^4*Po*log(rho)-4*kappa*log(rc)*rho^2*Po*m*cos(2*upsilon)-rho^4*Po+rho^4*Pcr-4*rho^2*Po*m^2+4*rho^2*Pcr*m^2)/(-2*m*rho^2*cos(2*upsilon)+rho^4+m^2)^(1/2)/rho/G/log(rc);
			# Ut(i,j)  = -1/4*R*eta*m*sin(2*upsilon)*(2*Pcr*rho^2*log(rc)-kappa*rho^2*Pcr+rho^2*kappa*Po+2*kappa*log(rc)*Pcr*rho^2-rho^2*Po+rho^2*Pcr-2*Pcr+2*Po-4*rho^2*Pcr*log(rho)+4*rho^2*Po*log(rho))/(-2*m*rho^2*cos(2*upsilon)+rho^4+m^2)^(1/2)/rho/G/log(rc);
		end
	end

	Z=Rho.*exp.(im*alpha);
	X=real(1/2*(Z+m./Z));
	Y=imag(1/2*(Z+m./Z));
	#println(X)

	grid_length = 50

	x = range(-2, 2, length=grid_length)
	y = range(-2, 2, length=grid_length)

	xgrid = x
	ygrid = y

	#itp = interpolate(X,Y, Srr, Gridded(Linear()))
	
	nx = println(length(X))
	nx = println(length(X))

	grid_x = kron(ones(grid_length),x')
	grid_y = kron(y,ones(1,grid_length))

	#println(typeof(vec(X)))

	# grid_x = kron(ones(length(vec(X)),vec(X)'))
	# grid_y = kron(vec(Y),ones(1,length(vec(Y))))

	# Perform the interpolation
	
	np = 200 
	xmin = -2.
	xmax = 4.
	ymin = -2.
	ymax = 4.
	
	x = xmin .+ xmax.*rand(np)
	y = ymin .+ ymax.*rand(np)

	println(typeof(x))
	println(typeof(vec(X)))
	println(length(vec(X)))
	println(length(vec(Y)))

	points = [x y]
	println(typeof(points))
	println(size(points))

	points = [vec(X), vec(Y)]

	points = permutedims(hcat(points...))'

	#reduce(vcat,transpose.(points))
	println(typeof(points))
	println(size(points))

	val = vec(Srr)

	grid_val = si.griddata(points,val,(grid_x,grid_y),method="nearest")

	# p = Plots.scatter(vec(X), vec(Y), zcolor=vec(Srr), title="Interpolated heatmap")
	# Plots.heatmap!(p, xgrid, ygrid, grid_val, title="Interpolated heatmap")

	p = Plots.heatmap(xgrid, ygrid, grid_val, title="Interpolated heatmap")

	#p = Plots.scatter(X, Y, Srr, title="Interpolated heatmap")

	#p = Plots.heatmap(xgrid, ygrid, grid_val, title="Interpolated heatmap")

return p
	#p = Plots.heatmap(xgrid, ygrid, grid_val, title="Interpolated heatmap")
	
	# # Some 2D function
	# f(x,y) = sin(x)*cos(y)
	#
	# # Location of random points to sample the function at
	# np = 200 
	# xmin = -2.
	# xmax = 4.
	# ymin = -2.
	# ymax = 4.
	# x = xmin .+ xmax.*rand(np)
	# y = ymin .+ ymax.*rand(np)
	# points = [x y]
	#
	# z = @. sin(x) * cos(y')
	#
	# # Value of the function at the random points
	# val = zeros(np)
	# for ip = 1:np
	# 	val[ip] = f(x[ip],y[ip])
	# end
	#
	# # Create a uniform grid to interpolate onto
	# nx = 50
	# ny = 75
	# xgrid = collect(range(xmin,xmax,nx))
	# ygrid = collect(range(ymin,ymax,ny))
	# grid_x = kron(ones(ny),xgrid')
	# grid_y = kron(ygrid,ones(1,nx))
	#
	# # Perform the interpolation
	# grid_val = si.griddata(points,val,(grid_x,grid_y),method="cubic")
	#
	# # x = range(-2, 2, length=10)
	# # y = range(-2, 2, length=10)
	# # z = @. cos(x) + sin(y')
	# #
	# # # Interpolation object (caches coefficients and such)
	# # itp = LinearInterpolation((x, y), z)
	# #
	# # # Fine grid
	# # x2 = range(extrema(x)..., length=300)
	# # y2 = range(extrema(y)..., length=200)
	# #
	# # # Interpolate
	# # z2 = [itp(x,y) for y in y2, x in x2]
	#
	#
	# xgrid = collect(range(xmin,xmax,nx))
	# ygrid = collect(range(ymin,ymax,ny))
	# # println(size(xgrid))	
	# # println(size(grid_val))	
	#
	# println(size(xgrid))
	# println(size(ygrid))
	# println(size(grid_val))
	#
	# p = Plots.heatmap(xgrid, ygrid, grid_val, title="Interpolated heatmap")
	#Plots.scatter!(p, [x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data", clim=(-2,2))

	#p = Plots.heatmap(x2, y2, z2, clim=(-2,2), title="Interpolated heatmap")
	#Plots.scatter!(p, [x for _ in y for x in x], [y for y in y for _ in x], zcolor=z[:]; lab="original data", clim=(-2,2))

end
