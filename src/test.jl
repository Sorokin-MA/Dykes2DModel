include("dykes_init.jl")
#include("dykes_funcs.jl")

function kernel_2(a)
    i = threadIdx().x
    a[i] += 1
    return
end

a = CUDA.zeros(1024)

@cuda threads=length(a) kernel_2(a)

@printf("\ndone\n")
