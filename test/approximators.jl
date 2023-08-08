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
using TotalLeastSquares

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
    device = gpu_device()

    # Define models
    rng = Random.Xoshiro(99)
    model = Chain(
        Dense(dims_in=>layers[1],tanh),
        Dense(layers[1]=>dims_out,identity)
    )
    parameters, state = Lux.setup(rng, model) .|> device
    xtrain = xtrain .|> device
    ytrain = ytrain .|> device

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
    savefig("test/approx_test/single_output.svg")
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
    savefig("test/approx_test/multi_output.svg")
end

abstract type AbstractSimpleHeuristic end

struct Uniform <: AbstractSimpleHeuristic end

function (un::Uniform)(xsample1, ysample1, xsample2, ysample2)
    N = size(xsample1,2)
    return (1/N)*ones(N)
end 

struct DiscreteDerivative <: AbstractSimpleHeuristic end 
function (dd::DiscreteDerivative)(xsample1, ysample1, xsample2, ysample2)
    num = ysample2 .- ysample1 
    den = xsample2 .- xsample1
    ρ  = map(norm, eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)) .+ 1e-8)
    return ρ
end 

struct DiscreteFourierDerivative <: AbstractSimpleHeuristic end
function (dd::DiscreteFourierDerivative)(xsample1, ysample1, xsample2, ysample2)
    num = fftshift(fft(ysample2 .- ysample1)) 
    den = fftshift(fft(xsample2 .- xsample1))
    ρ  = map(norm, eachslice(num, dims=2)) ./ (map(norm, eachslice(den, dims=2)) .+ 1e-8)
    return ρ
end 

struct SamplingNN <: AbstractApproximator
    rng
    dims_in::Int
    dims_out::Int
    layers::Vector
    density_resolution::Float64
    s1::Float64
    s2::Float64

    W::AbstractArray
    b::AbstractArray
    activation

    function SamplingNN(dims_in, dims_out, layers, s1, s2, res=2, activation=tanh)
        rng = Xoshiro(0)
        W = zeros(layers[1], dims_in)
        b = zeros(layers[1])
        new(rng, dims_in, dims_out, layers, res, s1, s2, W, b, activation)
    end 
end

function (snn::SamplingNN)(xtrain, ytrain, heuristic::typeof(AbstractSimpleHeuristic))
    _, m = size(xtrain)
    num_neurons = snn.layers[1]
    res = snn.density_resolution

    # Evaluate sampling density
    ρ_res = floor(Int,res*snn.layers[1])
    idxs_1 = sample(snn.rng, 1:m, ρ_res, replace=true)
    idxs_2 = sample(snn.rng, 1:m, ρ_res, replace=true)
    xsample1 = xtrain[:,idxs_1]
    ysample1 = ytrain[:,idxs_1]
    xsample2 = xtrain[:,idxs_2]
    ysample2 = ytrain[:,idxs_2]
    H = heuristic()
    ρ = H(xsample1, ysample1, xsample2, ysample2)

    # Assemble weights
    eval_idxs_1 = sample(snn.rng, 1:m, Weights(ρ), num_neurons, replace=true)
    eval_idxs_2 = sample(snn.rng, 1:m, Weights(ρ), num_neurons, replace=true)
    temp = xtrain[:,eval_idxs_1]' - xtrain[:,eval_idxs_2]'
    snn.W .= snn.s1 .* (temp / ((norm(temp) + 1e-8))^2)
    snn.b .= (snn.s2 .+ snn.W .* xtrain[:,eval_idxs_1]')[:]

    # Setup linear layer
    ϕ = snn.activation.(snn.W*xtrain .+ snn.b)

    # Total least squares
    @show cond(ϕ')
    W1 = (pinv(ϕ')*ytrain')'

    # Model
    model = x-> W1*(snn.activation.(snn.W * x .+ snn.b))
    return model
end

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