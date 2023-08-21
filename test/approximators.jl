using Lux
using Plots
using Random
using Optimisers
using Zygote
using Statistics
using LinearAlgebra
using BenchmarkTools
using FFTW
using StatsBase

abstract type AbstractApproximator end

struct NN <: AbstractApproximator
    dims_in
    dims_out
    layers
    nepochs
    learning_rate

    function NN(dims_in, dims_out, layers, nepochs, learning_rate)
        new(dims_in, dims_out, layers, nepochs, learning_rate)
    end 
end 

function (nn::NN)(xtrain::AbstractArray, ytrain::AbstractArray)

    # Get device
    # device =  gpu_device()

    # Define models
    rng = Random.Xoshiro(99)
    model = Chain(
        Dense(dims_in=>layers[1],tanh),
        Dense(layers[1]=>dims_out,identity)
    )
    parameters, state = Lux.setup(rng, model) #.|> device
    xtrain = xtrain #.|> device
    ytrain = ytrain #.|> device

    # Evaluate once
    val, st  = model(xtrain, parameters, state)

    # Loss function
    function loss(parameters, state)
        ypred, lstate = model(xtrain, parameters, state)
        loss = mean((ytrain .- ypred).^2)
        return loss, lstate
    end

    # Training loop
    function train(loss,learning_rate, nepochs, parameters, state) 
        optim = Optimisers.Adam(learning_rate)
        optim_state = Optimisers.setup(optim, parameters) 
        loss_history = []
        for epoch in 1:nepochs
            (l, state,), back = Zygote.pullback(loss, parameters, state)
            grad, _ = back((1.0,nothing))
            optim_state, parameters = Optimisers.update(optim_state, parameters, grad)
            push!(loss_history, l)
            println("Epoch $epoch \t Loss $l")
        end 
        return parameters, state, loss_history
    end

    # For now manual
    parameters, state, loss_history = train(loss,learning_rate, nepochs, parameters, state)

    return (model, parameters, state), loss_history
end 

begin
    N = 100
    x = LinRange(0.0,2π,N)
    y = sin.(x)
    dims_in = dims_out = 1
    xtrain = reshape(x,(dims_in,:))
    ytrain = reshape(y,(dims_out,:))
    layers = [100]

    learning_rate = 1e-2
    nepochs = 30000
    nn = NN(dims_in, dims_out, layers, nepochs, learning_rate)
    (model, ps, st), loss_history = nn(xtrain, ytrain)

    ypred = model(xtrain, ps, st)[1]
    f1 = plot(xtrain[:], ytrain[:], gridwidth=2.0,label="Ground truth")
    scatter!(xtrain[:], ypred[:], ms=2.0, label="Prediction from network", title="Sine function optimized with ADAM")

    f2 = plot(loss_history,yaxis=:log,title="Loss evolution",xlabel="Iterations", ylabel="Loss (log)", label=false)
    plot(f1,f2,size=(700,500))
    savefig("RandomFeaturePotentials/test/approx_test/single_output.svg")
end

begin
    N = 100
    x = LinRange(0.0,2π,N)
    y = [sin.(4*x),cos.(4*x),sin.(4*x).*cos.(4*x)]
    y = reduce(hcat,y)'
    nepochs = 60000
    dims_in = 1
    dims_out = 3
    layers = [100]
    xtrain = reshape(x,(dims_in,:))
    ytrain = reshape(y,(dims_out,:))
    nn = NN(dims_in, dims_out, layers, nepochs, learning_rate)
    (model, ps, st), loss_history = nn(xtrain, ytrain)

    ypred = model(xtrain, ps, st)[1]
    f1 = plot(xtrain[:], ytrain', gridwidth=2.0,label="Ground truth")
    scatter!(xtrain[:], ypred', ms=2.0, label="Prediction from network", title="Sine function optimized with ADAM", legend=:outertop)

    f2 = plot(loss_history,yaxis=:log,title="Loss evolution",xlabel="Iterations", ylabel="Loss (log)", label=false)
    plot(f1,f2,size=(1000,500))
    savefig("RandomFeaturePotentials/test/approx_test/multi_output.svg")
end

# Models for the features in the neural network -- W,b
abstract type AbstractFeatureModel end

struct LinearFeatureModel <: AbstractFeatureModel 
    s1::Float64
    s2::Float64
    W::Any
    b::Any

    function LinearFeatureModel(s1,s2)
        new(s1,s2,nothing,nothing)
    end
end

function (model::LinearFeatureModel)(rng, xtrain, idxs, ρ, K)
    W(x1,x2) = model.s1*(x1-x2)/norm(x1-x2)
    b(x1,x2) = ((x1-x2)/norm(x1-x2))' * x1 + model.s2
    
    M = size(xtrain)[end]
    nsamples = K
    idx_from = wsample(rng, idxs, nsamples, Weights(ρ), replace=false)
    idx_to = wsample(rng, idxs, nsamples, Weights(ρ), replace=false)
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
    idx = sample(1:M, nsamples)
    ρ = (1/nsamples)*ones(nsamples)
    return idx, ρ
end 

function (heuristic::FiniteDifference)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples)
    idx_to = sample(1:M, nsamples)
    num = ytrain[:,idx_to] .- ytrain[:,idx_from]
    den = xtrain[:,idx_to] .- xtrain[:,idx_from]
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) / (map(norm, eachslice(den, dims=2))+ϵ)
    return idx_from,ρ
end

function (heuristic::FullDerivative)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    num = diff(ytrain,dims=2)
    den = diff(xtrain,dims=2)
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) / (map(norm, eachslice(den, dims=2))+ϵ)
    idx = 1:M-1
    return idx,ρ
end 

function (heuristic::RandomDerivative)(xtrain, ytrain, Nl, multiplicity)
    M = size(xtrain)[end]
    nsamples = Nl*multiplicity
    idx_from = sample(1:M, nsamples)
    idx_to = mod.(idx_from .+ 1,M)
    num = ytrain[:,idx_to] .- ytrain[:,idx_from]
    den = xtrain[:,idx_to] .- xtrain[:,idx_from]
    ϵ = 1e-8
    ρ = map(norm, eachslice(num, dims=2)) / (map(norm, eachslice(den, dims=2))+ϵ)
    return idx_from,ρ
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

    function SamplingNN(dims_in, dims_out, layers, feature_model; multiplicity=2, activation=tanh)
        rng = Xoshiro(0)
        new(rng, dims_in, dims_out, layers, multiplicity, feature_model, activation)
    end 
end

function (snn::SamplingNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ;atol=1e-12,optimize=false)
    # Evaluate sampling density
    M = size(xtrain)[end]
    Nl = snn.layers[1]
    H = heuristic()
    idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)

    # Sample weights and biases
    W,b = snn.feature_model(snn.rng, xtrain, idxs, ρ, Nl)

    # Solve for coefficients of last layer
    bases = activation.(W*xtrain .+ b)'
    if M<1000 || ~optimize
        coeff = pinv(bases,atol=λ)*ytrain
    else
        coeff,stats = Krylov.lslq(bases,ytrain,λ=λ,atol=atol)
    end  # size(coeff) = K, output_dims 

    # Setup model
    model = x -> (coeff' * activation.(W1*x .+ b1))

    return model
end


N = 100
x = LinRange(0.0,2π,N)
y = sin.(x)
dims_in = dims_out = 1
layers = [20]
res = 1.0
xtrain = reshape(x,(dims_in,:))
ytrain = reshape(y,(dims_out,:))
heuristic = Uniform
λ = 1e-12
s2 = log(1.5)
s1 = 2*s2
f_model = LinearFeatureModel(s1,s2)
snn = SamplingNN(dims_in,dims_out,layers,f_model,multiplicity=5,activation=tanh)
model = snn(xtrain, ytrain, heuristic, λ,atol=1e-9,optimize=false)


begin
    N = 100
    x = LinRange(0.0,2π,N)
    y = sin.(x)
    s1 = 1.0
    s2 = 1.0
    dims_in = dims_out = 1
    layers = [20]
    res = 1.0
    snn = SamplingNN(dims_in, dims_out, layers, s1, s2,res) 
    xtrain = reshape(x,(dims_in,:))
    ytrain = reshape(y,(dims_out,:))
    heuristic = DiscreteDerivative
    model = snn(xtrain, ytrain, heuristic)

    ypred = model(xtrain)
    f1 = plot(xtrain[:], ytrain[:], gridwidth=2.0,label="Ground truth")
    scatter!(xtrain[:], ypred[:], ms=2.0, label="Prediction from network", title="Sampling neural network")

    f2 = plot(1:size(ytrain,2),ytrain[:]-ypred[:],title="Error",label=false,gridwidth=2.0)

    fig = plot(f1,f2,size=(700,500))
    savefig("test/approx_test/single_sampling.svg")
    savefig("test/approx_test/test.png")
end 

begin
    N = 100
    x = LinRange(0.0,2π,N)
    y = sin.(2*x)
    s1 = 1.0
    s2 = 1.0
    dims_in = dims_out = 1
    layers = [20]
    res = 1.0
    snn = SamplingNN(dims_in, dims_out, layers, s1, s2,res) 
    xtrain = reshape(x,(dims_in,:))
    ytrain = reshape(y,(dims_out,:))
    heuristic = DiscreteFourierDerivative
    model = snn(xtrain, ytrain, heuristic)

    ypred = model(xtrain)
    f1 = plot(xtrain[:], ytrain[:], gridwidth=2.0,label="Ground truth")
    scatter!(xtrain[:], ypred[:], ms=2.0, label="Prediction from network", title="Sampling neural network")
    savefig("test/approx_test/single_sampling.svg")
    savefig("test/approx_test/test.png")
end 


begin
    N = 100
    x = LinRange(0.0,2π,N)
    y = sin.(2*x)
    s1 = 1.0
    s2 = 1.0
    dims_in = dims_out = 1
    layers = [20]
    res = 1.0
    snn = SamplingNN(dims_in, dims_out, layers, s1, s2,res) 
    xtrain = reshape(x,(dims_in,:))
    ytrain = reshape(y,(dims_out,:))
    heuristic = Uniform
    model = snn(xtrain, ytrain, heuristic)

    ypred = model(xtrain)
    f1 = plot(xtrain[:], ytrain[:], gridwidth=2.0,label="Ground truth")
    scatter!(xtrain[:], ypred[:], ms=2.0, label="Prediction from network", title="Sampling neural network")
    savefig("test/approx_test/single_sampling.svg")
    savefig("test/approx_test/test.png")
end 


begin
    N = 100
    x = LinRange(0.0,2π,N)
    y = [sin.(4*x),cos.(4*x),sin.(4*x).*cos.(4*x)]
    y = reduce(hcat,y)'
    dims_in = 1
    dims_out = 3
    layers = [200]
    xtrain = reshape(x,(dims_in,:))
    ytrain = reshape(y,(dims_out,:))
    s1 = 1.0
    s2 = 0.0
    res = 1.0

    snn = SamplingNN(dims_in, dims_out, layers, s1, s2, res)
    model = snn(xtrain, ytrain, DiscreteDerivative)
    ypred = model(xtrain)
    f1 = plot(xtrain[:], ytrain', gridwidth=2.0,label="Ground truth")
    scatter!(xtrain[:], ypred', ms=2.0, label="Prediction from network", title="Sampling neural network", legend=:outertop)
    
    f2 = plot(1:size(ypred,2),(ypred-ytrain)', title="Error", gridwidth=2.0,label=false)
    plot(f1,f2, size=(1000,500))
    savefig("test/approx_test/multi_sampling.svg")
    savefig("test/approx_test/test.png")
end