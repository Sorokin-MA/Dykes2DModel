@with_kw mutable struct GridParams
	critVol::Array{Float64,1} = Array{Float64,1}(undef,0);
	ndikes::Array{Int32,1} = Array{Int32,1}(undef,0);
	particle_edges::Array{Int32,1} = Array{Int32,1}(undef,0);
	marker_edges::Array{Int32,1} = Array{Int32,1}(undef, 0);
	T::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);
	T_old::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);
	C::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);
	wts::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);
	pcnt::CuArray{Int32,1} = CuArray{Int32,1}(undef, 0);

	a::CuArray{Float64} = CuArray{Float64}(undef, (1, 2));

	px::CuArray{Float64} = CuArray{Float64}(undef, 0);	#x coordinate of particle
	py::CuArray{Float64} = CuArray{Float64}(undef, 0); #y coordinate of particle
	pT::CuArray{Float64} = CuArray{Float64}(undef, 0);	#Temperature of particle
	pPh::CuArray{Int8} = CuArray{Int8}(undef, 0);		#???#Ph?


	px_dikes::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);	#x of dykes particles
	py_dikes::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);	#y of dykes particles
	

	mx::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);		#x of marker
	my::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);		#y of marker
	mT::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);		#T of marker

	#???
	staging::Array{Float64,1} = Array{Float64,1}(undef, 0);
	npartcl_d::CuArray{Int32,1} = CuArray{Int32,1}(undef, 1);
	npartcl_h::Array{Int32,1} = Array{Int32,1}(undef, 1);

	#small grid itself
	L::CuArray{Int32,1} = CuArray{Int32,1}(undef, 0);

	#small grid on host
	L_host::Array{Int32,1} = Array{Int32,1}(undef, 0);

	#???
	mfl::CuArray{Float64,1} = CuArray{Float64,1}(undef, 0);


	#a and b of ellips for dikes
	dike_a::Array{Float64,1} = Array{Float64,1}(undef, 0);
	dike_b::Array{Float64,1} = Array{Float64,1}(undef, 0);

	#x and y coordinate of center
	dike_x::Array{Float64,1} = Array{Float64,1}(undef, 0);
	dike_y::Array{Float64,1} = Array{Float64,1}(undef, 0);

	#???
	dike_t::Array{Float64,1} = Array{Float64,1}(undef, 0);

	h_px = Array{Float64,1}(undef, 0);
	h_py = Array{Float64,1}(undef, 0);

	h_px_dikes = Array{Float64,1}(undef, 0);
	h_py_dikes = Array{Float64,1}(undef, 0);

	h_mx = Array{Float64,1}(undef, 0);
	h_my = Array{Float64,1}(undef, 0);
	h_mT = Array{Float64,1}(undef, 0);

	eruptionSteps = Vector{Int32}();
end

@with_kw mutable struct VarParams
	Lx::Float64 = 0.0				#X size of researched area (m)
	Ly::Float64 = 0.0				#Y size of researched area (m)
	lam_r_rhoCp::Float64 = 0.0		#Thermal conductivity of rock/(density*specific heat capacity)
	lam_m_rhoCp::Float64 = 0.0		#Thermal conductivity of magma/(density*specific heat capacity)
	L_Cp::Float64 = 0.0			#???#dT/Ste, Ste = dT/(Lheat/Cp); L_heat/Cp
	T_top::Float64 = 0.0				#Temperature on the top of area (°C)
	T_bot::Float64 = 0.0				#Temperature at the bottom of the area (°C)
	T_magma::Float64 = 0.0			#Magma instrusion temperature(°C)
	tsh::Float64 = 0.0			#???#nondimensonal 0.75
	gamma::Float64 = 0.0			#???#nondimensional 0.1
	Ly_eruption::Float64 = 0.0	#???#2000, m
	nu::Float64 = 0.0				#Poisson ratio of rock
	G::Float64 = 0.0				#???#E/(2*(1+nu));
	d::Float64 = 0.0			#???#time step
	dx::Float64 = 0.0		#X dimension step
	dy::Float64 = 0.0		#Y dimension step
	dt::Float64 = 0.0		#Y dimension step
	eiter::Float64 = 0.0			#???#epsilon
	pic_amount::Float64 = 0.0	#???#0.05

	pmlt::Int32 = 0			#???#unused
	nx::Int32 = 0					#Resolution for X dimension
	ny::Int32 = 0					#Resolution for Y dimension
	nxl::Int32 = 0					#
	nyl::Int32 = 0					#
	nl::Int32 = 0					#?
	nt::Int32 = 0					#?
	niter::Int32 = 0				#?
	nout::Int32 = 0				#?
	nsub::Int32 = 0				#?
	nerupt::Int32 = 0				#how often check for eruptions
	npartcl::Int32 = 0				#number of particles
	nmarker::Int32 = 0				#number of markers
	nSample::Int32 = 0				#size of a Sample, 1000

	idike::Int32 = 0				#number of dikes
	npartcl0::Int32 = 0				#initial amount of particles

	max_npartcl::Int32 = 0				#size of a Sample, 1000
	max_nmarker::Int32 = 0

	is_eruption::Bool = false
end
