
function d2dm_test()
	return
end


function dykes_muskh()
	return
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
