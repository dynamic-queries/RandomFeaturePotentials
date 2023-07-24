abstract type AbstractDescriptor end

struct SimpleSimilarityDescriptor <: AbstractDescriptor
    X::AbstractArray
    D::AbstractArray
    
    function SimpleSimilarityDescriptor(X, Z, kernel, params)
        R = pairwise_displacements(X)
        ker = kernel(R,Z,params)
        k = ker()
        new(X,k)
    end 
end 

function Base.write(des::SimpleSimilarityDescriptor, filename::String)
    npzwrite(filename, des.D)
end 

function augment(X, k)
    _, m = size(k)
    D = zeros(m,m,4)
    for i=1:m
        for j=1:m
            D[i,j,:] = [k[i,j], (k[i,j]^2)*X[i,1], (k[i,j]^2)*X[i,2], (k[i,j]^2)*X[i,3]]
        end 
    end 
    return reshape(D,(m,:))
end 

struct AugmentedSimilarityDescriptor <: AbstractDescriptor
    X::AbstractArray
    D::AbstractArray

    function AugmentedSimilarityDescriptor(X, Z, kernel, params)
        R = pairwise_displacements(X)
        ker = kernel(R,Z,params)
        k = ker()
        R = augment(X, k)
        new(X,R)
    end 
end

function Base.write(des::AugmentedSimilarityDescriptor, filename::String)
    npzwrite(filename, des.D)
end 