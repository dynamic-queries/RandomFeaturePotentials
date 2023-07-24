begin
    X = rand(100,3)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    params = [α, σ]
    kernel = RBFKernel
    des = SimpleSimilarityDescriptor(X, Z, kernel, params)
    @show size(des.D)
    write(des, "write_test/simple_rbf.npy")
end

begin
    X = rand(100,3)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    params = [α]
    kernel = GDMLKernel
    des = SimpleSimilarityDescriptor(X, Z, kernel, params)
    @show size(des.D)
    write(des, "write_test/simple_gdml.npy")
end

begin
    X = rand(100,3)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    params = [α]
    kernel = CoulombMatrix
    des = SimpleSimilarityDescriptor(X, Z, kernel, params)
    @show size(des.D)
    write(des, "write_test/simple_cm.npy")
end


begin
    X = rand(100,3)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    params = [α, σ]
    kernel = RBFKernel
    des = AugmentedSimilarityDescriptor(X, Z, kernel, params)
    @show size(des.D)
    write(des, "write_test/aug_rbf.npy")
end

begin
    X = rand(100,3)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    params = [α]
    kernel = GDMLKernel
    des = SimpleSimilarityDescriptor(X, Z, kernel, params)
    @show size(des.D)
    write(des, "write_test/aug_gdml.npy")
end

begin
    X = rand(100,3)
    Z = rand(1:10, 100)
    α = 0.5
    σ = 0.1
    params = [α]
    kernel = CoulombMatrix
    des = SimpleSimilarityDescriptor(X, Z, kernel, params)
    @show size(des.D)
    write(des, "write_test/aug_cm.npy")
end
