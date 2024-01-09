include("../src/dykes_init.jl")
include("../src/dykes_funcs.jl")
include("../dykes_test_funcs.jl")

using Test

@testset "Funcs" begin
	@testset "mf_magma" begin
		@test true
	end;
end;
