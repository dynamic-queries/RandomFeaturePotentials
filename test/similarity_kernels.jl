begin
    x = reshape(LinRange(0.0,1.0,100),(:,1))
    R = pairwise_displacements(x)
    Z = rand(1:1, 100)
    σ = 0.05
    rbf = RBFKernel(R,Z,σ)
    k = rbf()
    display(heatmap(Array(k)))
end 

begin
    x = rand(100,3)
    R = pairwise_displacements(x)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    rbf = RBFKernel(R,Z,σ,α)
    k = rbf()
    display(heatmap(Array(k)))
end

begin
    x = reshape(LinRange(0.0,1.0,100),(:,1))
    R = pairwise_displacements(x)
    Z = rand(1:1, 100)
    cm = CoulombMatrix(R,Z)
    k = cm()
    display(heatmap(Array(k)))
end 

begin
    x = rand(100,3)
    R = pairwise_displacements(x)
    Z = rand(1:10, 100)
    α = 0.5
    cm = RBFKernel(R,Z,α)
    k = cm()
    display(heatmap(Array(k)))
end

begin
    x = reshape(LinRange(0.0,1.0,100),(:,1))
    R = pairwise_displacements(x)
    Z = rand(1:1, 100)
    gk = GDMLKernel(R,Z)
    k = gk()
    display(heatmap(Array(k)))
end 

begin
    x = rand(100,3)
    R = pairwise_displacements(x)
    Z = rand(1:10, 100)
    α = 0.5
    gk = GDMLKernel(R,Z,α)
    k = gk()
    display(heatmap(Array(k)))
end

using LinearAlgebra
using MAT
using Plots
using Pkg
using NPZ
Pkg.activate("RandomFeaturePotentials")
using RandomFeaturePotentials

abstract type AbstractSimilarityKernel end
abstract type GraphKernel <: AbstractSimilarityKernel end

struct DiffusionMap <: GraphKernel
    R::AbstractArray
    Z::AbstractArray
    σ::Float64
end 

function (Dmap::DiffusionMap)()
    # Assemble Gram matrix
    _, m, d = size(Dmap.R)
    gaussian(r) = exp((-r/Dmap.σ)^2)
    k = zeros(m,m)
    for i=1:m
        for j=1:m
            k[i,j] = gaussian(norm(Dmap.R[i,j,:]))
        end 
    end 

    # Normalize Gram matrix
    incidence = vec(sum(k,dims=2))
    Incidence = diagm(1 ./ incidence)

    # Kernel matrix
    Q = Incidence*k*Incidence
    D = diagm(vec(sum(Q,dims=2)))
    normalized_Q = D^(-1/2)*Q*D^(-1/2)

    # Evaluate the diffusion coordinates
    eigvals, eigvecs = eigen(normalized_Q)
    return eigvals, eigvecs
end

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


# QM7 data

data = matread("data/QM7/qm7.mat")
R = data["R"]
r = pairwise_displacements(R[10,:,:])
z = ones(23)

begin
    σ = 5.0
    dmap = DiffusionMap(r,z,σ)
     diffusion_coordinates = dmap()
    f1 = plot(diffusion_coordinates,legend=false, title="Diffusion coordinates")
end

begin
    α = 1.0
    cm = CoulombMatrix(r,z,α)
    k = cm()
    f2 = plot(k,legend=false, title="Coulomb matrix coordinates")
end 


begin
    α = 5.0
    cm = RBFKernel(r,z,α)
    k = cm()
    f3 = plot(k,legend=false, title="Radial bases coordinates")
end 

plot(f1,f2,f3,layout=(1,3),size=(1200,300))


# GDML data
data = "data/MD17/md17_benzene2017"

begin 
    R = npzread(string(data,"/R.npy"))[rand(1:1000),:,:]
    r = pairwise_displacements(R)
    z = npzread(string(data, "/z.npy"))

    begin
        σ = 5.0
        dmap = DiffusionMap(r,z,σ)
         diffusion_coordinates = dmap()
        f1 = plot(diffusion_coordinates,legend=false, title="Diffusion coordinates");
    end

    begin
        α = 1.0
        cm = CoulombMatrix(r,z,α)
        k = cm()
        f2 = plot(k,legend=false, title="Coulomb matrix coordinates");
    end 


    g1 = plot(f1,f2,layout=(1,3),size=(1200,300))

    R = npzread(string(data,"/R.npy"))[rand(1:1000),:,:]
    r = pairwise_displacements(R)
    z = npzread(string(data, "/z.npy"))

    begin
        σ = 5.0
        dmap = DiffusionMap(r,z,σ)
         diffusion_coordinates = dmap()
        f1 = plot(diffusion_coordinates,legend=false, title="Diffusion coordinates");
    end

    begin
        α = 1.0
        cm = CoulombMatrix(r,z,α)
        k = cm()
        f2 = plot(k,legend=false, title="Coulomb matrix coordinates");
    end 

    g2 = plot(f1,f2,layout=(1,3),size=(1200,300))

    plot(g1,g2)
    savefig("RandomFeaturePotentials/test/cm_vs_i")
end