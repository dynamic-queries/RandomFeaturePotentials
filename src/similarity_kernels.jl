function pairwise_displacements(X::AbstractArray)
    m,d = size(X)
    R = zeros(m,m,d)
    for i=1:m
        for j=1:m
            R[i,j,:] = X[i,:] - X[j,:]
        end 
    end 
    return R
end 

abstract type AbstractSimilarityKernel end
abstract type GraphKernel <: AbstractSimilarityKernel end
abstract type HypergraphKernel <: AbstractSimilarityKernel end

struct RBFKernel <: GraphKernel
    R::AbstractArray
    Z::AbstractArray
    σ::Float64
    α::Float64
    f::Function

    function RBFKernel(R,Z,σ,α=0.24)
        f = (r,z1,z2) ->  (z1*z2)^α * exp(-((r)/(σ))^2)
        new(R,Z,σ,α,f)
    end

    function RBFKernel(R,Z,params::Vector)
        RBFKernel(R,Z,params...)
    end 
end 

function (rbf::RBFKernel)()
    _, m, d = size(rbf.R)
    k = zeros(m,m)
    for i=1:m
        for j=1:m
            k[i,j] = rbf.f(norm(rbf.R[i,j,:]), rbf.Z[i], rbf.Z[j])
        end 
    end 
    return k
end 

struct CoulombMatrix <: GraphKernel
    R::AbstractArray
    Z::AbstractArray
    α::Float64
    f::Function
    
    function CoulombMatrix(R,Z,α=0.24)
        f = (r,z1,z2) -> (z1*z2)^α *(!(r<1e-4) ? (1/r) : 1.0)
        new(R,Z,α,f)
    end
    
    function CoulombMatrix(R, Z, params::Vector)
        CoulombMatrix(R,Z,params...)
    end 
end 

function (cm::CoulombMatrix)()
    _, m, d = size(cm.R)
    k = zeros(m,m)
    for i=1:m
        for j=1:m
            k[i,j] = cm.f(norm(cm.R[i,j,:]), cm.Z[i], cm.Z[j])
        end 
    end 
    return k
end 

struct GDMLKernel <: GraphKernel
    R::AbstractArray
    Z::AbstractArray
    α::Float64
    f::Function
    
    function GDMLKernel(R,Z,α=0.24)
        f = (r,z1,z2) -> (z1*z2)^α * (!(r<1e-4) ? (1/r) : 1.0)
        new(R,Z,α,f)
    end

    function GDMLKernel(R,Z,params::Vector)
        GDMLKernel(R,Z,params...)
    end 
end 

function (gk::GDMLKernel)()
    _, m, d = size(gk.R)
    k = zeros(m,m)
    for i=1:m
        for j=1:m
            k[i,j] = gk.f(norm(gk.R[i,j,:]), gk.Z[i], gk.Z[j])
        end 
    end 
    return k
end 

struct DiffusionMap <: GraphKernel
    R::AbstractArray
    Z::AbstractArray
    σ::Float64    
end 


function (Dmap::DiffusionMap)()
    
    # Assemble Hankel matrix
    _, m, d = size(Dmap.R)
    gaussian(r) = exp((-norm(r)/Dmap.σ)^2)
    k = zeros(m,m)
    for i=1:m
        for j=1:m
            k[i,j] = gaussian(Dmap.R[i,j,:])
        end 
    end 

    # Normalize Hankel matrix
    incidence = vec(sum(k,dims=2))
    Incidence = diagm(1 ./ incidence)

    # Kernel matrix
    Q = Incidence*k*Incidence
    D = diagm(vec(sum(Q,dims=2)))
    normalized_Q = D^(-1/2)*Q*D^(-1/2)

    # Evaluate the diffusion coordinates
    eigvals, eigvecs = eigen(normalized_Q)
    return eigvecs
end

struct HigherOrderDiffusionMap <: HypergraphKernel

end 

struct HypergraphLaplacian <: HypergraphKernel

end 
