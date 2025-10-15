using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse

include("../conditional_test.jl")

# Random seed
Random.seed!(11)

# Input
nx = 5
ny = 11
n_channel = 3
batchsize = 10
in_split, split_num = InvertibleNetworks.ConditionalLayerCorrelation_splitdims(n_channel)

TT = Float64
X = randn(TT, nx, ny, n_channel, batchsize)
X0 = randn(TT, nx, ny, n_channel, batchsize)
Y0 = randn(TT, nx, ny, n_channel, batchsize)
dX = X - X0

# Test activation functions.
name = "Cosh"
@testset verbose = true "$name" begin
    println("Testing $name")

    Sm = InvertibleNetworks.Cosh(X)
    ΔSm = randn(TT, size(Sm))
    ΔX = InvertibleNetworks.CoshGrad(ΔSm, X)

    Sm2, back = Flux.pullback(InvertibleNetworks.Cosh, deepcopy(X))
    @test norm(Sm - Sm2) == 0
    ΔX2, = back(ΔSm)
    @test norm(ΔX - ΔX2) < 1f-6
end

name = "Sinh"
@testset verbose = true "$name" begin
    println("Testing $name")

    Sm = InvertibleNetworks.Sinh(X)
    ΔSm = randn(TT, size(Sm))
    ΔX = InvertibleNetworks.SinhGrad(ΔSm, Sm)

    Sm2, back = Flux.pullback(InvertibleNetworks.Sinh, deepcopy(X))
    @test norm(Sm - Sm2) == 0
    ΔX2, = back(ΔSm)
    @test norm(ΔX - ΔX2) < 2f-5
end

name = "DampedCosh"
@testset verbose = true "$name" begin
    println("Testing $name")

    Sm = InvertibleNetworks.DampedCosh(X)
    ΔSm = randn(TT, size(Sm))
    ΔX = InvertibleNetworks.DampedCoshGrad(ΔSm, X)

    Sm2, back = Flux.pullback(InvertibleNetworks.DampedCosh, deepcopy(X))
    @test norm(Sm - Sm2) == 0
    ΔX2, = back(ΔSm)
    @test norm(ΔX - ΔX2) < 2f-5
end

name = "DampedSinh"
@testset verbose = true "$name" begin
    println("Testing $name")

    Sm = InvertibleNetworks.DampedSinh(X)
    ΔSm = randn(TT, size(Sm))
    ΔX = InvertibleNetworks.DampedSinhGrad(ΔSm, Sm)

    Sm2, back = Flux.pullback(InvertibleNetworks.DampedSinh, deepcopy(X))
    @test norm(Sm - Sm2) == 0
    ΔX2, = back(ΔSm)
    @test norm(ΔX - ΔX2) < 2f-5
end

@testset verbose = true "ConditionalLayerCorrelation (n_channel_cond=$n_channel_cond)" for n_channel_cond in [n_channel, n_channel+2]
    Cond = randn(TT, nx, ny, n_channel_cond, batchsize)

    # Test with the simplest configuration.
    name = "ConditionalLayerCorrelation with no subnetworks"
    @testset verbose = true "$name (scale_activation=$scale_activation)" for scale_activation in ["damped_cosh", "softplus"], shift_cond_scalar in [true, false]
        @testset verbose = true "$name (shift_activation=$shift_activation)" for shift_activation in ["damped_sinh", "softplus"]
            @testset verbose = true "$name (shift_cond_scalar=$shift_cond_scalar)" for shift_cond_scalar in [true, false]
                @testset verbose = true "$name (out_chan=$out_chan)" for out_chan in [split_num, 2*split_num]
                    println("Testing $name")
                    layer_conv1x1 = nothing
                    layer_constant = LayerConstant(glorot_uniform(nx, ny, out_chan))
                    if scale_activation == "damped_cosh"
                        scale_activation = DampedCoshLayer()
                    elseif scale_activation == "softplus"
                        scale_activation = SoftplusLayer()
                    end
                    if shift_activation == "damped_sinh"
                        shift_activation = DampedSinhLayer()
                    elseif shift_activation == "softplus"
                        shift_activation = SoftplusLayer()
                    end
                    L = ConditionalLayerCorrelation(nothing, layer_constant; logdet=true, shift_cond_scalar=true, scale_activation, shift_activation)

                    if TT != Float32
                        forward(Float32.(X), Float32.(Cond), L)
                        P = deepcopy(get_params(L))
                        for p in P
                            if isnothing(p.data)
                                continue
                            end
                            p.data = TT.(p.data)
                        end
                    else
                        forward(X, Cond, L)
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
                        dp.data ./= norm(p.data) + eps(TT)
                    end
                    set_params!(L, deepcopy(P))

                    conditional_layer_test_inverse(L, X, Cond, dX)
                    conditional_layer_test_gradient(L, P, dP, X, Cond, dX; name, do_flux=true)
                end
            end
        end
    end

    # Test Conv1x1
    name = "Conv1x1NoMutate"
    @testset verbose=true "$name" begin
        println("Testing $name")
        layer_conv1x1 = Conv1x1(n_channel; logdet=true)
        layer_conv1x1_nomutate = Conv1x1NoMutate(n_channel; logdet=true)

        P = deepcopy(get_params(layer_conv1x1))
        if TT != Float32
            for p in P
                p.data = TT.(p.data)
            end
            set_params!(layer_conv1x1, deepcopy(P))
        end
        set_params!(layer_conv1x1_nomutate, deepcopy(P))

        Y, logdet = forward(X, layer_conv1x1)
        Y_f, logdet = forward(X, layer_conv1x1_nomutate)

        @test norm(Y - Y_f) ./ norm(Y) < 2f-6

        ΔY = randn(eltype(Y), size(Y))
        ΔX = inverse((ΔY, Y), layer_conv1x1)[1]
        ΔP = deepcopy(get_grads(layer_conv1x1))

        forward_test = function (X, P)
            set_params!(layer_conv1x1_nomutate, P)
            Y, logdet = forward(X, layer_conv1x1_nomutate)
            Y
        end
        Y_f, back = Flux.pullback(forward_test, deepcopy(X), deepcopy(P))

        @test norm(Y - Y_f) ./ norm(Y) < 2f-6

        ΔX_f, ΔP_fT = back(ΔY)
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        @test norm(ΔX - ΔX_f) ./ norm(ΔX) < 2f-6
        @test norm(ΔP - ΔP_f) ./ norm(ΔP) < 2f-6
    end

    # Test with Conv1x1NoMutate.
    name = "ConditionalLayerCorrelation with Conv1x1NoMutate"
    @testset verbose=true "$name" begin
        println("Testing $name")
        layer_conv1x1 = Conv1x1NoMutate(n_channel; logdet=true)
        layer_constant = LayerConstant(glorot_uniform(nx, ny, split_num))
        L = ConditionalLayerCorrelation(layer_conv1x1, layer_constant; logdet=true)

        if TT != Float32
            forward(Float32.(X), Float32.(Cond), L)
            P = deepcopy(get_params(L))
            for p in P
                p.data = TT.(p.data)
            end
            set_params!(L, deepcopy(P))
        else
            forward(X, Cond, L)
            P = deepcopy(get_params(L))
        end

        # Set up for parameters test.
        dP = deepcopy(P)
        for (p, dp) in zip(P, dP)
            dp.data = randn(eltype(p.data), size(p.data))
            dp.data ./= norm(p.data)
        end

        conditional_layer_test_inverse(L, X, Cond, dX)
        conditional_layer_test_gradient(L, P, dP, X, Cond, dX; name)
    end

    # Test with Conv1x1.
    name = "ConditionalLayerCorrelation with Conv1x1"
    @testset verbose=true "$name" begin
        println("Testing $name")
        layer_conv1x1 = Conv1x1(n_channel; logdet=true)
        layer_constant = LayerConstant(glorot_uniform(nx, ny, split_num))
        L = ConditionalLayerCorrelation(layer_conv1x1, layer_constant; logdet=true)

        if TT != Float32
            forward(Float32.(X), Float32.(Cond), L)
            P = deepcopy(get_params(L))
            for p in P
                p.data = TT.(p.data)
            end
            set_params!(L, deepcopy(P))
        else
            forward(X, Cond, L)
            P = deepcopy(get_params(L))
        end

        # Set up for parameters test.
        dP = deepcopy(P)
        for (p, dp) in zip(P, dP)
            dp.data = randn(eltype(p.data), size(p.data))
            dp.data ./= norm(p.data)
        end

        conditional_layer_test_inverse(L, X, Cond, dX)
        conditional_layer_test_gradient(L, P, dP, X, Cond, dX; name)
    end

    name = "ConditionalLayerCorrelation with ResidualBlock"
    @testset verbose=true "$name (out_chan=$out_chan)" for out_chan in [split_num, 2 * split_num]
        println("Testing $name")
        k1 = 3
        k2 = 3
        p1 = 1
        p2 = 1
        fan = true
        n_hidden = 4
        activation = SoftplusLayer()
        final_activation = IdentityActivation()
        layer_resblock = ResidualBlock(in_split+n_channel_cond, n_hidden; n_out=out_chan, k1, k2, p1, p2, fan, activation, final_activation)
        L = ConditionalLayerCorrelation(nothing, layer_resblock; logdet=true)

        if TT != Float32
            forward(Float32.(X), Float32.(Cond), L)
            P = deepcopy(get_params(L))
            for p in P
                p.data = TT.(p.data)
            end
            set_params!(L, deepcopy(P))
        else
            forward(X, Cond, L)
            P = deepcopy(get_params(L))
        end

        # Set up for parameters test.
        dP = deepcopy(P)
        for (p, dp) in zip(P, dP)
            dp.data = randn(eltype(p.data), size(p.data))
            a = norm(p.data)
            if a != 0
                dp.data ./= a
            end
        end

        conditional_layer_test_inverse(L, X, Cond, dX)
        conditional_layer_test_gradient(L, P, dP, X, Cond, dX; name)
    end


    name = "ConditionalLayerCorrelation with ResidualBlock and Conv1x1NoMutate"
    @testset verbose=true "$name (out_chan=$out_chan)" for out_chan in [split_num, 2 * split_num]
        println("Testing $name")
        k1 = 3
        k2 = 3
        p1 = 1
        p2 = 1
        fan = true
        n_hidden = 4
        layer_conv1x1 = Conv1x1NoMutate(n_channel; logdet=true)
        activation = SoftplusLayer()
        final_activation = IdentityActivation()
        layer_resblock = ResidualBlock(in_split+n_channel_cond, n_hidden; n_out=out_chan, k1, k2, p1, p2, fan, activation, final_activation)
        L = ConditionalLayerCorrelation(layer_conv1x1, layer_resblock; logdet=true)

        if TT != Float32
            forward(Float32.(X), Float32.(Cond), L)
            P = deepcopy(get_params(L))
            for p in P
                p.data = TT.(p.data)
            end
            set_params!(L, deepcopy(P))
        else
            forward(X, Cond, L)
            P = deepcopy(get_params(L))
        end

        # Set up for parameters test.
        dP = deepcopy(P)
        for (p, dp) in zip(P, dP)
            dp.data = randn(eltype(p.data), size(p.data))
            a = norm(p.data)
            if a != 0
                dp.data ./= a
            end
        end

        conditional_layer_test_inverse(L, X, Cond, dX)
        conditional_layer_test_gradient(L, P, dP, X, Cond, dX; name)
    end
end
