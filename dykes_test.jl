include("dykes_funcs.jl")

using Test

@testset "Funcs" begin
    @testset "mf_magma" begin
           @test mf_magma(0) == 10
       end;
end;
