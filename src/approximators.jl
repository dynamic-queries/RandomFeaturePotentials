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
        Dense(dims_in=>layers[1],relu),
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
    ρ_res = floor(Int,res*m)
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