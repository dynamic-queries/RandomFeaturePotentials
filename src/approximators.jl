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
    device = Lux.gpu_device()

    # Define models
    rng = Random.Xoshiro(99)
    model = Chain(
        Dense(nn.dims_in=>nn.layers[1],relu),
        Dense(nn.layers[1]=>nn.dims_out,identity)
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
    parameters, state, loss_history = train(loss,nn.learning_rate, nn.nepochs, parameters, state)

    return (model, parameters, state), loss_history
end 

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

function (snn::SamplingNN)(xtrain, ytrain, heuristic::typeof(AbstractHeuristic),λ;atol=1e-12,optimize=false)
    # Evaluate sampling density
    M = size(xtrain)[end]
    Nl = snn.layers[1]
    H = heuristic()
    idxs,ρ = H(xtrain, ytrain, Nl, snn.multiplicity)

    # Sample weights and biases
    W,b = snn.feature_model(snn.rng, xtrain, idxs, ρ, Nl)

    # Solve for coefficients of last layer
    bases = snn.activation.(W*xtrain .+ b)'
    @show cond(bases)
    if optimize==false
        coeff = pinv(bases,atol=λ)*ytrain'
    else
        r = size(ytrain,1)
        coeff,stats = Krylov.lslq(Array(bases),Array(ytrain'[:]),λ=λ,atol=atol)
        coeff = reshape(coeff,r,:)'
    end  # size(coeff) = K, output_dims 

    # Setup model
    model = x -> (coeff' * snn.activation.(snn.feature_model.W*x .+ snn.feature_model.b))

    return model
end
