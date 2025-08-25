"""
File in which gonna be different tests
"""

using Revise
using Dykes2DModel
using CUDA
using Plots
using Random

using Test

function add(a, b)
    return a + b
end

function test_mf_interpolation(a, b)
    return a + b
end


@testset "Basic Arithmetic" begin
    @test add(2, 3) == 5
    @test add(-1, 1) == 0
    @test add(0, 0) == 0
end

