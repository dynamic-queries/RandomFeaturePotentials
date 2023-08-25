using Lux
using Plots
using Random
using Optimisers
using Zygote
using Statistics
using Random
using LinearAlgebra
using ForwardDiff
using BenchmarkTools
using FFTW
using Optimization
using OptimizationOptimJL
using Krylov
using CUDA
using StatsBase
using BenchmarkTools
ENV["GKSwstype"] = "100"

abstract type AbstractApproximator end

abstract type AbstractFeatureModel end

mutable struct LinearFeatureModel <: AbstractFeatureModel 
    s1::Float64
    s2::Float64
    W::Any
    b::Any

    function LinearFeatureModel(s1,s2)
        new(s1,s2,nothing,nothing)
    end
end

function (model::LinearFeatureModel)(rng, xtrain, idxs, ρ, K)
    W(x1,x2) = model.s1*(x1-x2)/norm(x1-x2).^2
    b(x1,x2) = ((x1-x2)/(norm(x1-x2)).^2)' * x1 + model.s2
    
    nsamples = K
    L = length(ρ)
    idx = wsample(rng, 1:L, Weights(ρ), nsamples, replace=false)
    idx_from = idxs[1][idx]
    idx_to = idxs[2][idx]
    W1 = []
    b1 = []
    for i=1:K
        k1 = xtrain[:,idx_from[i]]
        k2 = xtrain[:,idx_to[i]]
        if k1!=k2
            push!(W1,W(k1,k2))
            push!(b1,b(k1,k2))
        end
    end 
    model.W = reduce(hcat, W1)'
    model.b = b1
    return model.W, model.b
end

# Heuristics for sampling density
abstract type AbstractHeuristic end
struct Uniform <: AbstractHeuristic end
struct FiniteDifference <: AbstractHeuristic end
struct FullDerivative <: AbstractHeuristic end
struct RandomDerivative <: AbstractHeuristic end

function (heuristic::Uniform)(xtrain,  ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples,replace=true)
    idx_to = sample(1:M, nsamples,replace=true)
    ρ = (1/nsamples)*ones(nsamples)
    return [idx_from, idx_to], ρ
end 

function (heuristic::FiniteDifference)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples, replace=true)
    idx_to = sample(1:M, nsamples, replace=true)
    num = ytrain[:,idx_to] .- ytrain[:,idx_from]
    den = xtrain[:,idx_to] .- xtrain[:,idx_from]
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)).+ϵ)
    return [idx_from, idx_to],ρ
end

function (heuristic::FullDerivative)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    num = diff(ytrain,dims=2)
    den = diff(xtrain,dims=2)
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)).+ϵ)
    idx1 = 1:M-1
    idx2 = 2:M
    return [idx1, idx2],ρ
end 

function (heuristic::RandomDerivative)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples, replace=true)
    idx_to = mod.(idx_from .+ 1, M).+1
    num = ytrain[:,idx_to] .- ytrain[:,idx_from]
    den = xtrain[:,idx_to] .- xtrain[:,idx_from]
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)) .+ϵ)
    return [idx_from, idx_to],ρ
end 

# 
struct SamplingNN <: AbstractApproximator
    rng
    dims_in::Int
    dims_out::Int
    layers::Vector
    multiplicity::Int
    feature_model::AbstractFeatureModel
    activation

    function SamplingNN(dims_in, dims_out, layers, feature_model; multiplicity=1, activation=tanh)
        rng = Xoshiro(0)
        new(rng, dims_in, dims_out, layers, multiplicity, feature_model, activation)
    end 
end

function (snn::SamplingNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ;atol=1e-12,optimize=false,CustomArray=Array)
    # Evaluate sampling density
    M = size(xtrain)[end]
    Nl = snn.layers[1]
    H = heuristic()
    idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)

    # Sample weights and biases
    W,b = snn.feature_model(snn.rng, xtrain, idxs, ρ, Nl)

    W = CustomArray(W)
    b = CustomArray(b)
    # Solve for coefficients of last layer
    bases = CustomArray(snn.activation.(W*xtrain .+ b)')
    @show cond(bases)
    if M<1000 || ~optimize
        coeff = pinv(bases,atol=λ)*CustomArray(ytrain')
    else
        coeff,stats = Krylov.lsqr(Array(bases),Array(ytrain')[:],λ=λ,atol=atol)
    end  # size(coeff) = K, output_dims 

    # Setup model
    model = x -> (coeff' * snn.activation.(W*x .+ b))
    return model
end

# # Examples
# begin
#     begin # Preamble
#         N = 100
#         x = LinRange(0.0,2π,N)
#         y = sin.(4*x)
#         dims_in = dims_out = 1
#         layers = [20]
#         res = 1.0
#         xtrain = reshape(x,(dims_in,:))
#         ytrain = reshape(y,(dims_out,:))
#         λ = 1e-12
#         s2 = log(1.5)
#         s1 = 2*s2
#         f_model = LinearFeatureModel(s1,s2)

#         snn = SamplingNN(dims_in,dims_out,layers,f_model,multiplicity=2,activation=tanh)
#         M = size(xtrain)[end]
#         Nl = snn.layers[1]
#         nothing
#     end 

#     begin
#         heuristic = Uniform
#         H = heuristic()
#         idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)
#         f1 = plot(ρ,ylabel="ρ",title="Uniform sampling",legend=false,gridlinewidth=2.0,linewidth=0.5)
#         display(f1)
#         savefig("test/approx_test/Uniform.svg")

#         heuristic = FiniteDifference
#         H = heuristic()
#         idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)
#         f1 = plot(ρ,ylabel="ρ",title="FD",legend=false,gridlinewidth=2.0,linewidth=0.5)
#         display(f1)
#         savefig("test/approx_test/FD.svg")


#         heuristic = FullDerivative
#         H = heuristic()
#         idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)
#         f1 = plot(ρ,ylabel="ρ",title="Derivative",legend=false,gridlinewidth=2.0,linewidth=0.5)
#         display(f1)
#         savefig("test/approx_test/FullDerivative.svg")


#         heuristic = RandomDerivative
#         H = heuristic()
#         idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)
#         f1 = plot(ρ,ylabel="ρ",title="Random sampled derivative",legend=false,gridlinewidth=2.0,linewidth=0.5)
#         display(f1)
#         savefig("test/approx_test/RandomDerivative.svg")
#     end

#     begin
#         layers = [20]
#         λ = 1e-15
#         heuristic = Uniform
#         model = snn(xtrain,ytrain,heuristic,λ,atol=1e-8,optimize=false)
#         test_data = xtrain
#         ypred = model(test_data)
#         f1 = plot(xtrain[:], ytrain[:], gridwidth=2.0,label="Ground truth")
#         scatter!(xtrain[:], ypred[:], ms=2.0, label="Prediction from network", title="Sampling neural network - US")

#         f2 = plot(1:size(ytrain,2),ytrain[:]-ypred[:],title="Error",label=false,gridwidth=2.0)

#         fig = plot(f1,f2,size=(1000,300))
#         display(fig)
#         savefig("test/approx_test/Uniform_single_sampling.svg")
#         savefig("test/approx_test/test.png")
#     end

#     begin
#         layers = [20]
#         λ = 1e-15
#         heuristic = FiniteDifference
#         model = snn(xtrain,ytrain,heuristic,λ,atol=1e-8,optimize=false)
#         test_data = xtrain
#         ypred = model(test_data)
#         f1 = plot(xtrain[:], ytrain[:], gridwidth=2.0,label="Ground truth")
#         scatter!(xtrain[:], ypred[:], ms=2.0, label="Prediction from network", title="Sampling neural network - FD")

#         f2 = plot(1:size(ytrain,2),ytrain[:]-ypred[:],title="Error",label=false,gridwidth=2.0)

#         fig = plot(f1,f2,size=(1000,300))
#         display(fig)
#         savefig("test/approx_test/FiniteDifference_single_sampling.svg")
#         savefig("test/approx_test/test.png")
#     end
# end 


# begin
#     begin # Preamble
#         N = 100
#         x = LinRange(0.0,2π,N)
#         y = sin.(4*x)
#         z = sin.(2*x) .* cos.(4*x)

#         dims_in = 1
#         dims_out = 2
#         layers = [100]
#         res = 1.0
#         xtrain = reshape(x,(dims_in,:))
#         ytrain = reshape(hcat(y,z)',(dims_out,:))
#         λ = 1e-12
#         s2 = log(1.5)
#         s1 = 2*s2
#         f_model = LinearFeatureModel(s1,s2)

#         snn = SamplingNN(dims_in,dims_out,layers,f_model,multiplicity=5,activation=tanh)
#         M = size(xtrain)[end]
#         Nl = snn.layers[1]
#         nothing
#     end 

#     begin
#         heuristic = Uniform
#         H = heuristic()
#         idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)
#         f1 = plot(ρ,ylabel="ρ",title="Uniform sampling",legend=false,gridlinewidth=2.0,linewidth=0.5)
#         display(f1)
#         savefig("test/approx_test/Uniform.svg")

#         heuristic = FiniteDifference
#         H = heuristic()
#         idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)
#         f1 = plot(ρ,ylabel="ρ",title="FD",legend=false,gridlinewidth=2.0,linewidth=0.5)
#         display(f1)
#         savefig("test/approx_test/FD.svg")
#     end

#     begin
#         λ = 1e-14
#         heuristic = Uniform
#         model = snn(xtrain,ytrain,heuristic,λ,atol=1e-14,optimize=false)
#         test_data = xtrain
#         ypred = model(test_data)
#         f1 = plot(xtrain[:], ytrain[1,:], gridwidth=2.0,label="Ground truth")
#         scatter!(xtrain[:], ypred[1,:], ms=2.0, label="Prediction from network", title="Sampling neural network - US")
#         plot!(xtrain[:], ytrain[2,:], gridwidth=2.0,legend=false)
#         scatter!(xtrain[:], ypred[2,:], ms=2.0)


#         f2 = plot(1:length(ytrain[:]),ytrain[:]-ypred[:],title="Error",label=false,gridwidth=2.0)

#         fig = plot(f1,f2,size=(1000,300))
#         display(fig)
#         savefig("test/approx_test/Uniform_multiple_sampling.svg")
#         savefig("test/approx_test/test.png")
#     end

#     begin
#         λ = 1e-14
#         heuristic = FiniteDifference
#         model = snn(xtrain,ytrain,heuristic,λ,atol=1e-8,optimize=false)
#         test_data = xtrain
#         ypred = model(test_data)
#         f1 = plot(xtrain[:], ytrain[1,:], gridwidth=2.0,label="Ground truth")
#         scatter!(xtrain[:], ypred[1,:], ms=2.0, label="Prediction from network", title="Sampling neural network - FD")
#         plot!(xtrain[:], ytrain[2,:], gridwidth=2.0,legend=false)
#         scatter!(xtrain[:], ypred[2,:], ms=2.0)

#         f2 = plot(1:length(ytrain[:]),ytrain[:]-ypred[:],title="Error",label=false,gridwidth=2.0)

#         fig = plot(f1,f2,size=(1000,300))
#         display(fig)
#         savefig("test/approx_test/FiniteDifference_multiple_sampling.svg")
#         savefig("test/approx_test/test.png")
#     end
# end 


# Test GDB dataset
using MAT
filename = "../data/QM7/qm7.mat"
file = matread(filename)
C = reshape(file["X"],(7165,:))'
E = file["T"]

dims_in = 529
dims_out = 1
layers = [5000]
res = 1
λ = 1e-12
s2 = log(1.5)
s1 = 2*s2
f_model = LinearFeatureModel(s1,s2)
snn = SamplingNN(dims_in,dims_out,layers,f_model,multiplicity=res,activation=tanh)
heuristic = Uniform

@time model1 = snn(C,E,heuristic,Float32(λ),atol=Float32(1e-8),optimize=true)
# @btime model2  = snn(C,E,heuristic,λ,atol=1e-8,optimize=true)

Epred = model1(C)
f1 = plot(E[:],E[:], gridwidth=2.0,label="Ground truth")
scatter!(E[:], Epred[:], ms=2.0, label="Prediction from network", title="Sampling neural network - US")
savefig("test/molecule_approx/E_QM7.svg")
savefig("test/molecule_approx/test.png")