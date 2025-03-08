module MechanicalKernel

using CUDA
using Interpolations
using Adapt

dykes_crystalinity = 1 .- Vector{Float64}([1, 0.9978, 0.9955, 0.9918, 0.9881, 0.9401, 0.9273, 0.9231, 0.9189, 0.9029, 0.8859, 0.8532, 0.7972, 0.6777, 0.371, 0.2195, 0.1545, 0.1184, 0.1111, 0.1038, 0.0965, 0.0892, 0.0819, 0.0769, 0.0721, 0.0672, 0.0624, 0.0573, 0.0516, 0.0459, 0.0402, 0.0338, 0.0265, 0.0192, 0.0143, 0.0143, 0.0143, 0.0143, 0.0121, 0.0099, 0.0076, 0.006, 0.0056, 0.0051, 0.0047, 0.0043, 0.0037, 0.0028, 0.0019, 0.0009, 0])
dykes_temp = Vector{Float64}([699.2355, 708.9755, 718.7156, 728.4557, 738.1957, 747.9358, 757.6758, 767.4159, 777.156, 786.896, 796.6361, 806.3761, 816.1162, 825.8563, 835.5963, 845.3364, 855.0765, 864.8165, 874.5566, 884.2966, 894.0367, 903.7768, 913.5168, 923.2569, 932.9969, 942.737, 952.4771, 962.2171, 971.9572, 981.6972, 991.4373, 1001.1774, 1010.9174, 1020.6575, 1030.3976, 1040.1376, 1049.8777, 1059.6177, 1069.3578, 1079.0979, 1088.8379, 1098.578, 1108.318, 1118.0581, 1127.7982, 1137.5382, 1147.2783, 1157.0183, 1166.7584, 1176.4985, 1186.2385])

A_x = range(699.2355, 1186.2385, 51);
itp = interpolate(dykes_crystalinity, BSpline(Cubic(Line(OnGrid()))))
itp = Interpolations.scale(itp, A_x)
itp = extrapolate(itp, Flat())

cuitp = adapt(CuArray{eltype(dykes_temp)}, itp);


function mf_campi_rhyolite(T)
    return itp(T)
end


function dmf_rhyolite(T)
    t1 = T * T
    t9 = exp(0.961026e3 - 0.186618e-5 * t1 * T + t1 * 0.447948e-2 + T * (-0.359050e1))
    t12 = (0.1e1 + t9) * (0.1e1 + t9)
    return 0.559856e-5 / t12 * t9 * (t1 - 0.160022e4 * T + 0.641326e6)
end

function dmf_basalt(T)
    t1 = T * T
    t11 = exp(0.143636887899999948e3 - 0.2214446257e-6 * t1 * T + t1 * 0.572468110399999928e-3 + T * (-0.494427718499999891e0))
    t14 = (0.1e1 + t11)^0.2e1
    return 0.6643338771e-6 * (t1 - 0.1723434948e4 * T + 0.7442458310e6) * t11 / t14
end

function mf_rhyolite(T)
    t2 = T * T
    t7 = exp(0.961026371384066e3 - 0.3590508961e1 * T + 0.4479483398e-2 * t2 - 0.1866187556e-5 * t2 * T)
    return 0.1e1 / (0.1e1 + t7)
end

function mf_basalt(T)
    t2 = T * T
    t7 = exp(960 - 3.554 * T + 0.4468e-2 * t2 - 1.907e-06 * t2 * T)
    return 0.1e1 / (0.1e1 + t7)
end

#coefficient which involved in heat equasion
function dmf_magma(T)
    return dmf_basalt(T)
end

#coefficient which involved in heat equasion
function dmf_rock(T)
    return only.(Interpolations.gradient.(Ref(cuitp), T))
end

#melt fraction of magma
function mf_magma(T)
    return mf_basalt(T)
end

#melt fraction of host rocks
function mf_rock(T)
    return mf_campi_rhyolite(T)
end



end
