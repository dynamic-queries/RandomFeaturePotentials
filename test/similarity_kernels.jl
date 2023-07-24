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