using RandomFeaturePotentials
using Test
using Plots

@testset "RandomFeaturePotentials.jl" begin
    include("similarity_kernels.jl")
    include("descriptors.jl")
end
