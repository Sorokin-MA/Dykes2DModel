module D2DM_GUI_Interpolations
    export CUITP, CPUITP, init_interpolation!, get_interpolation, get_cpu_interpolation
    
    using Interpolations, Adapt, CUDA
    
    const CUITP = Ref{Any}(nothing)     # GPU version
    const CPUITP = Ref{Any}(nothing)    # CPU version
    
    function init_interpolation!()
#=
        dykes_crystalinity = Float64[1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0]
        
        A_x = range(699.2355, 1186.2385, 51)
        itp = interpolate(dykes_crystalinity, BSpline(Cubic(Natural(OnGrid()))))
        itp = Interpolations.scale(itp, A_x)
        itp = extrapolate(itp, Flat())
        =#

		dykes_crystalinity = 1 .- Vector{Float64}([1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0])
		dykes_temp = Vector{Float64}([699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385])

		#d2dm_campi_rhyolite::Interpolations.MonotonicInterpolation = interpolate(dykes_temp, dykes_crystalinity, SteffenMonotonicInterpolation())

		A_x = range(699.2355, 1186.2385, 51);
		dykes_crystalinity_ = range(699.2355, 1186.2385, 51);
		nodes = (A_x,)
		itp = interpolate(dykes_crystalinity, BSpline(Cubic(Natural(OnGrid()))))
		itp = Interpolations.scale(itp, A_x)
		itp = extrapolate(itp, Flat())

        
        # Store both versions
        CPUITP[] = itp                                    # CPU version
        CUITP[] = adapt(CuArray{Float64}, itp)           # GPU version
        
        return itp  # Return the CPU version
    end
    
    function get_interpolation()
        return CUITP[]  # Return GPU version
    end
    
    function get_cpu_interpolation()
        return CPUITP[]  # Return CPU version
    end
end