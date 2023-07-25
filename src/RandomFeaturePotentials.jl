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
    

    include("similarity_kernels.jl")
    include("descriptors.jl")
    include("invariates.jl")
    include("approximators.jl")
    include("benchmarks.jl")

    export pairwise_displacements
    export RBFKernel, GDMLKernel, CoulombMatrix, AngularKernel 
    export SimpleSimilarityDescriptor, AugmentedSimilarityDescriptor, write
    export Eigenvalues, SingularValues, Eigenfunctions, PrincipalFunctions, DeepSet, CustomSet
    export normalize, rotate_me_not
    export make_set
    export NN, ANN, SNN, ASNN
end