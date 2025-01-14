using CoordRefSystems
using Unitful

function d2dm_test()
	return
end

function meshgrid(x, y)
	X = [i for i in x, j in 1:length(y)]
	Y = [j for i in 1:length(x), j in y]
	return [X, Y]
end

Base.@kwdef mutable struct DykeParam
	a::Float64   = 3;
	b::Float64   = 2;
	x::Float64   = 1;
	y::Float64   = 0.5;
	phi::Float64 = 0.8;
end

function calc_Sxx(point_x, point_y, m, Pcr, eta, rc, Po, x_move, y_move, turn)
	point_x = point_x - x_move;
	point_y = point_y - y_move;
	cartesian = Cartesian(point_x, point_y)
	polar = convert(CoordRefSystems.Polar, cartesian)
	rho = ustrip(u"m", polar.ρ)
	upsilon = Float64(polar.ϕ)

	upsilon = upsilon + turn;

	Z_real = rho*exp.(1im*upsilon);
	if(real(Z_real) >= 0)
	     Zr_rev_z = Z_real + sqrt(Z_real^2 - m);
	else
	     Zr_rev_z = Z_real - sqrt(Z_real^2 - m);
	end

	X=real(Zr_rev_z);
	Y=imag(Zr_rev_z);

	
	Rho = rho;
	alpha = upsilon;

	cartesian = Cartesian(X, Y)
	polar = convert(CoordRefSystems.Polar, cartesian)
	rho = ustrip(u"m", polar.ρ)
	upsilon = Float64(polar.ϕ)

	if(rho>=1)
	    Srr =  (eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32))+eta*rho^2*Pcr*m^4*log(1/(rc^8))+eta*Pcr*m^4*log(rho^8*rc^8)+eta*rho^6*Pcr*log(rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m*rho^6*Po*cos(2*upsilon)*log(rho^32)+eta*m^2*rho^2*Pcr*log(1/(rc^8))+eta*rho^2*Po*m^4*log(rc^8)+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+8*eta*Po*m^4+24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr-8*eta*rho^6*Pcr*m^2+8*eta*rho^6*Po*m^2-8*eta*Pcr*m^4-24*eta*m^2*rho^2*Po+eta*rho^6*Pcr*m^2*log(rc^8)+8*eta*rho^2*Pcr*m^4-8*eta*rho^2*Po*m^4+8*eta*rho^2*Pcr*m^3*cos(2*upsilon)-8*eta*m^3*Pcr*cos(2*upsilon)+8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))+eta*rho^6*Po*log(1/(rc^8))+24*eta*m*rho^4*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)-24*eta*m*rho^6*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-8*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)-24*eta*m*rho^4*Pcr*cos(2*upsilon)+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32)+24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*m^2*rho^2*Po*log(rc^8)+eta*rho^6*Po*m^2*log(1/(rc^8))+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/(rho^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
	    Stt = (eta*Pcr*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m^2*rho^2*Pcr*log(rc^8)+eta*rho^6*Po*m^2*log(rc^8)+eta*rho^2*Pcr*m^4*log(rc^8)+eta*m^2*rho^2*Po*log(1/(rc^8))+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+16*eta*Po*m^4-24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr+8*eta*rho^6*Pcr*m^2-8*eta*rho^6*Po*m^2-16*eta*Pcr*m^4+24*eta*m^2*rho^2*Po-8*eta*rho^2*Pcr*m^4+8*eta*rho^2*Po*m^4+56*eta*rho^2*Pcr*m^3*cos(2*upsilon)+8*eta*m^3*Pcr*cos(2*upsilon)-8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)+24*eta*m*rho^6*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-56*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)+24*eta*m*rho^4*Pcr*cos(2*upsilon)-24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*rho^6*Pcr*log(1/(rc^8))+eta*rho^6*Pcr*m^2*log(1/(rc^8))+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32*rc^32)+eta*rho^6*Po*log(rc^8)+eta*rho^2*Po*m^4*log(1/(rc^8))+eta*m*rho^6*Po*cos(2*upsilon)*log(1/rc^32*rho^32)+8*eta*rho^8*Pcr-8*eta*rho^8*Po+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/rho^32*rc^32)+eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
	    Srt = eta*m*sin(2*upsilon)*(-2*rho^6*Pcr*log(rc)+2*rho^2*log(rc)*Po*m^2-2*rho^4*log(rc)*Po*m^2+2*rho^4*Pcr*log(rc)*m^2+2*m*Pcr*rho^2*cos(2*upsilon)-2*m*Po*rho^2*cos(2*upsilon)-2*rho^4*Pcr*m^2+2*rho^6*log(rc)*Po+2*rho^4*Po*m^2-2*m*rho^4*Pcr*cos(2*upsilon)-m^2*Pcr+m^2*Po-3*rho^2*Po*m^2+3*rho^4*Po-3*rho^4*Pcr+3*rho^2*Pcr*m^2-3*rho^6*Po+3*rho^6*Pcr+2*m*rho^4*Po*cos(2*upsilon)+2*rho^4*Pcr*log(rc)-2*rho^2*Pcr*log(rc)*m^2-2*rho^4*log(rc)*Po)/log(rc)/(4*m^2*rho^4*cos(2*upsilon)^2+2*m^2*rho^4-4*rho^6*m*cos(2*upsilon)-4*rho^2*m^3*cos(2*upsilon)+rho^8+m^4);
	    Sxx = 1/2*(((-2*Rho.^2+1+Rho.^4).*Srr+(-2*Rho.^2-1-Rho.^4).*Stt).*cos(2*alpha)+(-2*Rho.^2+1+Rho.^4).*Srr+(2*Rho.^2+1+Rho.^4).*Stt+(-2*Rho.^4*sin(2*alpha)+2*sin(2*alpha)).*Srt)./(-2*Rho.^2*cos(2*alpha)+Rho.^4+1);
	else
	    Sxx = -1;
	end


	#println("Type of Sxx")
	#println(typeof(Sxx))
	return Sxx
end

function insert_dyke!(P, dyke_param::DykeParam, XX)
	#println(typeof(P))
	#TODO
	#0. Init fields
	#1. Calculate ro, phi fields
	#2. Calculate Srr, Stt, Srt
	#3. Calculate Sxx, Syy, Sxy
	#4. Calculate Eugen
	
	#0. Init
	#blockSize = (28, 32)
	#gridSize = (Int64(floor((nx + blockSize[1] - 1) / blockSize[1])), Int64(floor((ny + blockSize[2] - 1) / blockSize[2])))
	#@cuda blocks = gridSize[1], gridSize[2] threads = blockSize[1], blockSize[2] 
	
	#TODO:fix variables according to a and b
	
	r1::Float64 = 1;
	r2::Float64 = 2; # 1<=r2<=rc parameter which controls the plotted area
	m::Float64 = 0.75; #Variable in Joukovskiy equasion
	nu::Float64 = 0.3; #poussion coefficient
	eta::Float64 = (1-2*nu)/(1-nu)/2;
	rc::Float64 = 20; # rho_*
	Pcr::Float64 = 1; # Fluid pressure on cavity
	Po::Float64 = 0; # Fluid pressure on external boundary

	
	#1. ro, phi
	ro = r1:(r2-r1)/150:r2
	phi = 0:1/150:2*pi+pi/1e1

	Rho =zeros(length(ro),length(phi));
	alpha =zeros(length(ro),length(phi));

	X_rec, Y_rec = meshgrid(XX, XX)

	println(size(X_rec))

	Srr =zeros(length(ro),length(phi));
	Stt =zeros(length(ro),length(phi));
	Srt =zeros(length(ro),length(phi));

	#Sxy_new =zeros(length(X_rec));
	#Syy_new =zeros(length(X_rec));

	#2.
	for i in eachindex(ro)
		for j in eachindex(phi)
		Rho[i,j]=ro[i];
		rho = ro[i];
		alpha[i,j]=phi[j];
		upsilon=phi[j];
		Srr[i,j] =  (eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32))+eta*rho^2*Pcr*m^4*log(1/(rc^8))+eta*Pcr*m^4*log(rho^8*rc^8)+eta*rho^6*Pcr*log(rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m*rho^6*Po*cos(2*upsilon)*log(rho^32)+eta*m^2*rho^2*Pcr*log(1/(rc^8))+eta*rho^2*Po*m^4*log(rc^8)+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+8*eta*Po*m^4+24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr-8*eta*rho^6*Pcr*m^2+8*eta*rho^6*Po*m^2-8*eta*Pcr*m^4-24*eta*m^2*rho^2*Po+eta*rho^6*Pcr*m^2*log(rc^8)+8*eta*rho^2*Pcr*m^4-8*eta*rho^2*Po*m^4+8*eta*rho^2*Pcr*m^3*cos(2*upsilon)-8*eta*m^3*Pcr*cos(2*upsilon)+8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))+eta*rho^6*Po*log(1/(rc^8))+24*eta*m*rho^4*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)-24*eta*m*rho^6*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-8*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)-24*eta*m*rho^4*Pcr*cos(2*upsilon)+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32)+24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*m^2*rho^2*Po*log(rc^8)+eta*rho^6*Po*m^2*log(1/(rc^8))+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/(rho^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
		Stt[i,j]= (eta*Pcr*m^4*log(rho^8*rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m^2*rho^2*Pcr*log(rc^8)+eta*rho^6*Po*m^2*log(rc^8)+eta*rho^2*Pcr*m^4*log(rc^8)+eta*m^2*rho^2*Po*log(1/(rc^8))+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+16*eta*Po*m^4-24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr+8*eta*rho^6*Pcr*m^2-8*eta*rho^6*Po*m^2-16*eta*Pcr*m^4+24*eta*m^2*rho^2*Po-8*eta*rho^2*Pcr*m^4+8*eta*rho^2*Po*m^4+56*eta*rho^2*Pcr*m^3*cos(2*upsilon)+8*eta*m^3*Pcr*cos(2*upsilon)-8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))-24*eta*m*rho^4*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)+24*eta*m*rho^6*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-56*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)+24*eta*m*rho^4*Pcr*cos(2*upsilon)-24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*rho^6*Pcr*log(1/(rc^8))+eta*rho^6*Pcr*m^2*log(1/(rc^8))+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32*rc^32)+eta*rho^6*Po*log(rc^8)+eta*rho^2*Po*m^4*log(1/(rc^8))+eta*m*rho^6*Po*cos(2*upsilon)*log(1/rc^32*rho^32)+8*eta*rho^8*Pcr-8*eta*rho^8*Po+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/rho^32*rc^32)+eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32*rc^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
		Srt[i,j] = eta*m*sin(2*upsilon)*(-2*rho^6*Pcr*log(rc)+2*rho^2*log(rc)*Po*m^2-2*rho^4*log(rc)*Po*m^2+2*rho^4*Pcr*log(rc)*m^2+2*m*Pcr*rho^2*cos(2*upsilon)-2*m*Po*rho^2*cos(2*upsilon)-2*rho^4*Pcr*m^2+2*rho^6*log(rc)*Po+2*rho^4*Po*m^2-2*m*rho^4*Pcr*cos(2*upsilon)-m^2*Pcr+m^2*Po-3*rho^2*Po*m^2+3*rho^4*Po-3*rho^4*Pcr+3*rho^2*Pcr*m^2-3*rho^6*Po+3*rho^6*Pcr+2*m*rho^4*Po*cos(2*upsilon)+2*rho^4*Pcr*log(rc)-2*rho^2*Pcr*log(rc)*m^2-2*rho^4*log(rc)*Po)/log(rc)/(4*m^2*rho^4*cos(2*upsilon)^2+2*m^2*rho^4-4*rho^6*m*cos(2*upsilon)-4*rho^2*m^3*cos(2*upsilon)+rho^8+m^4);
		end
	end

	#T = Pcr-(Pcr-Po)*log(Rho)/log(rc);
	Z=Rho.*exp.(1im*alpha);
	Z_zh = 1/2*(Z+m./Z)
	X=real(Z_zh);
	Y=imag(Z_zh);


	#calc_Sxx(point_x, point_y, m, Pcr, eta, rc, Po, x_move, y_move, turn)


	#3.
	for i in eachindex(XX)
		for j in eachindex(XX)
	#		println("Type of P")
	#		println(typeof(P[i,j]))
			P[i,j] = P[i,j] + calc_Sxx(X_rec[i,j], Y_rec[i,j], m, Pcr, eta, rc, Po, dyke_param.x, dyke_param.y, dyke_param.phi);
		end
	end


	#4.


	return P
end



function d2dm_pres_test()
	#init phase
	dyke_param = DykeParam()
	
	#grid params
	Lx = 20000
	Ly = 20000

	nx = 2000
	ny = 2000

	dx::Float64 = Lx/(nx-1)
	dy::Float64 = Ly/(ny-1)

	xs = 0:dx:Lx
	ys = 0:dy:Ly

	#P::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);
	#P::Array{Float64,1} = Array{Float64,1}(undef, nx*ny)


	# kappa = 3-4*nu;
	# G = 1;
	# R=1;
	# C=1;


	XX = range(-3, 3, 200)
	X_rec, Y_rec = meshgrid(XX, XX)

	P = zeros(size(X_rec));
	#println(typeof(P))
	#println(size(X_rec))
	#println(size(P))
	insert_dyke!(P, dyke_param, XX)

	dyke_param = DykeParam()
	dyke_param.x = -1
	insert_dyke!(P, dyke_param, XX)
	#drawing phase
	#copyto!(h_P, P)

	#P_plot = PlotlyJS.heatmap(x = xs, y =ys, z = collect(eachrow(reshape(h_P, (nx, ny)))))
	#Plots.savefig(P_plot, "fig.png")
	#P_plot = Plots.heatmap(xs, ys, h_P)
	
	#title_string = "Preassure"
	#layout_inner = Layout(title = title_string)
	# P_plot = PlotlyJS.plot(PlotlyJS.heatmap(x = xs, y =ys, z = collect(eachrow(reshape(h_P, (nx, ny))))), layout_inner)
	# println(typeof(P_plot))
	# PlotlyJS.savefig(P_plot, "fig.png")
	
	P_plot = Plots.heatmap( XX,  XX,  P)
	println(typeof(P_plot))
	println(length(XX))
	println(length(P))
	Plots.savefig(P_plot, "fig.png")
	
	return
end


function dykes_muskh()
	return
end


# function calc_Srr(point_x, point_y, m, Pcr, eta, rc, Po)
# 	upsilon, rho = cart2pol(point_x, point_y);
# 	Z_real = rho*exp(1i*upsilon);
# 	if(point_x >= 0)
# 		Zr_rev_z = Z_real + sqrt(Z_real^2 -m);
# 	else
# 		Zr_rev_z = Z_real - sqrt(Z_real^2 -m);
# 	end
#
# 	X=real(Zr_rev_z);
# 	Y=imag(Zr_rev_z);
#
# 	upsilon, rho = cart2pol(X, Y);
# 	if(rho>=1)
# 		Srr =  (eta*rho^2*Pcr*m^3*cos(2*upsilon)*log(1/(rho^32))+eta*rho^2*Pcr*m^4*log(1/(rc^8))+eta*Pcr*m^4*log(rho^8*rc^8)+eta*rho^6*Pcr*log(rc^8)+eta*m^2*rho^4*Po*log(1/(rho^32))+eta*m^2*rho^4*Pcr*log(rho^32)+eta*m*rho^6*Po*cos(2*upsilon)*log(rho^32)+eta*m^2*rho^2*Pcr*log(1/(rc^8))+eta*rho^2*Po*m^4*log(rc^8)+eta*rho^8*Pcr*log(1/rc^8*rho^8)+eta*rho^8*Po*log(1/rho^8*rc^8)+8*eta*Po*m^4+24*eta*m^2*rho^2*Pcr+16*eta*m^2*rho^4*Po-16*eta*m^2*rho^4*Pcr-8*eta*rho^6*Pcr*m^2+8*eta*rho^6*Po*m^2-8*eta*Pcr*m^4-24*eta*m^2*rho^2*Po+eta*rho^6*Pcr*m^2*log(rc^8)+8*eta*rho^2*Pcr*m^4-8*eta*rho^2*Po*m^4+8*eta*rho^2*Pcr*m^3*cos(2*upsilon)-8*eta*m^3*Pcr*cos(2*upsilon)+8*eta*m^3*Po*cos(2*upsilon)+eta*Po*m^4*log(1/(rho^8*rc^8))+eta*rho^6*Po*log(1/(rc^8))+24*eta*m*rho^4*Po*cos(2*upsilon)-8*eta*m^2*rho^2*Po*cos(4*upsilon)+8*eta*m^2*rho^4*Po*cos(4*upsilon)-24*eta*m*rho^6*Po*cos(2*upsilon)+8*eta*m^2*rho^2*Pcr*cos(4*upsilon)-8*eta*m^2*rho^4*Pcr*cos(4*upsilon)-8*eta*rho^2*Po*m^3*cos(2*upsilon)+eta*m^2*rho^4*Pcr*log(rho^16)*cos(4*upsilon)+eta*m^2*rho^4*Po*log(1/(rho^16))*cos(4*upsilon)-24*eta*m*rho^4*Pcr*cos(2*upsilon)+eta*rho^2*Po*m^3*cos(2*upsilon)*log(rho^32)+24*eta*m*rho^6*Pcr*cos(2*upsilon)+eta*m^2*rho^2*Po*log(rc^8)+eta*rho^6*Po*m^2*log(1/(rc^8))+eta*m*rho^6*Pcr*cos(2*upsilon)*log(1/(rho^32)))/(m^2*rho^4*cos(4*upsilon)*log(rc^16)+m^2*rho^4*log(rc^32)+rho^6*m*cos(2*upsilon)*log(1/(rc^32))+rho^2*m^3*cos(2*upsilon)*log(1/(rc^32))+rho^8*log(rc^8)+m^4*log(rc^8));
# 	else
# 		Srr = -1;
# 	end
#
# 	return Srr
# end
