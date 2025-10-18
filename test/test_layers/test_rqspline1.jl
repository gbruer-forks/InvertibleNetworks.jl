using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse, RQSpline1

include("../invertible_test.jl")

# Random seed
Random.seed!(11)

# Input
nx = 5
ny = 11
n_channel = 3
batchsize = 10

TT = Float64
X = randn(TT, nx, ny, n_channel, batchsize)
X0 = randn(TT, nx, ny, n_channel, batchsize)
dX = X - X0

name = "RQSpline1"
println("Testing $name")
@testset verbose = true "$name no op" begin
    @testset verbose = true "$name (with_params=false, logdet=$logdet)" for logdet in [true, false]
        L = RQSpline1(; constrained_params=false, with_params=false, logdet)

        Y_logdet = forward(X, 0.5, 0.5, 1, L)
        if logdet
            Y, logdet = Y_logdet
            @test isapprox(norm(Y - X)/norm(X), 0f0; atol=1f-5)
            @test norm(logdet) == 0
        else
            Y = Y_logdet
            @test isapprox(norm(Y - X)/norm(X), 0f0; atol=1f-5)
        end

        X_inv = inverse(Y, 0.5, 0.5, 1, L)
        @test isapprox(norm(Y - X_inv)/norm(X), 0f0; atol=1f-5)

        ΔY = ones(size(Y))
        Δx, Δx0, Δy0, Δd, X_inv2 = backward(ΔY, Y, 0.5, 0.5, 1, L)
        @test isapprox(norm(X_inv2 - X_inv)/norm(X), 0f0; atol=1f-5)
        @test isapprox(norm(Δx - ΔY)/norm(X), 0f0; atol=1f-5)
        @test size(Δx0) == size(Y)
        @test size(Δy0) == size(Y)
        @test size(Δd) == size(Y)
    end

    @testset verbose = true "$name (with_params=true, logdet=$logdet)" for logdet in [true, false]
        L = RQSpline1(; with_params=true, logdet)

        Y_logdet = forward(X, L)
        if logdet
            Y, logdet = Y_logdet
            @test isapprox(norm(Y - X)/norm(X), 0f0; atol=1f-5)
            @test norm(logdet) == 0
        else
            Y = Y_logdet
            @test isapprox(norm(Y - X)/norm(X), 0f0; atol=1f-5)
        end

        X_inv = inverse(Y, L)
        @test isapprox(norm(Y - X_inv)/norm(X), 0f0; atol=1f-5)

        ΔY = ones(size(Y))
        Δx, X_inv2 = backward(ΔY, Y, L)
        @test isapprox(norm(X_inv2 - X_inv)/norm(X), 0f0; atol=1f-5)
        @test isapprox(norm(Δx - ΔY)/norm(X), 0f0; atol=1f-5)
        @test size(L.x0.grad) == size(Y)[1:end-1]
        @test size(L.y0.grad) == size(Y)[1:end-1]
        @test size(L.d.grad) == size(Y)[1:end-1]
    end
end

name = "RQSpline1"
println("Testing $name")
@testset verbose = true "$name full suite" begin
    @testset verbose = true "$name (random_init=$random_init)" for random_init in [false, true]
        L = RQSpline1(; constrained_params=true, with_params=true, logdet=true)

        if TT != Float32
            forward(Float32.(X), L)
            P = deepcopy(get_params(L))
            for p in P
                if isnothing(p.data)
                    continue
                end
                p.data = TT.(p.data)
            end
        else
            forward(X, L)
            P = deepcopy(get_params(L))
        end


        # Set up for parameters test.
        dP = deepcopy(P)
        for (p, dp) in zip(P, dP)
            if isnothing(p.data)
                p.data = [0]
                dp.data = [0]
                continue
            end
            dp.data = randn(eltype(p.data), size(p.data))
            dp.data ./= norm(p.data) + TT(1)
        end
        set_params!(L, deepcopy(P))
        if random_init
            for p in get_params(L)
                p.data = 5e-1 * randn(eltype(p.data), size(p.data))
            end
        end

        invertible_layer_test_inverse(L, X, dX)
        invertible_layer_test_gradient(L, P, dP, X, dX; name, do_flux=false, tol=1e-6)
    end
end
