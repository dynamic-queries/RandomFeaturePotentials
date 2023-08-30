module RandomFeaturePotentials
    using LinearAlgebra
    using NPZ
    using Lux
    using Optimisers
    using SparseArrays
    using LowRankApprox
    using Distributions
    using Statistics
    using StatsBase
    using Zygote
    using Random
    using RegularizationTools
    using Krylov
    using CUDA
    

    include("similarity_kernels.jl")
    include("descriptors.jl")
    include("invariates.jl")
    include("approximators.jl")
    include("benchmarks.jl")

    export pairwise_displacements
    export RBFKernel, GDMLKernel, CoulombMatrix, DiffusionMap
    export SimpleSimilarityDescriptor, AugmentedSimilarityDescriptor, write
    export Eigenvalues, SingularValues, Eigenfunctions, PrincipalFunctions, DeepSet, CustomSet
end