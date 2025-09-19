using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse

include("../grad_test.jl")

# Random seed
Random.seed!(11)

###################################################################################################
# Test invertibility

function conditional_layer_test_inverse(L, X, Cond, dX)
    X_ = inverse(forward(X, Cond, L)[1], Cond, L)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-5)

    X_ = forward(inverse(X, Cond, L), Cond, L)[1]
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-5)
end

function conditional_layer_test_gradient(L, P, dP, X, Cond, dX; name, do_flux=nothing)
    # Loss Function
    loss = function (L, P, X, Cond; with_grad)
        if !isnothing(P)
            set_params!(L, P)
        end
        Y, logdet = forward(X, Cond, L)
        f = log_likelihood(Y) - logdet
        if with_grad
            ΔY = ∇log_likelihood(Y)
            ΔX = backward(ΔY, Y, Cond, L)[1]
            return f, ΔX
        end
        return f
    end


    # Gradient test w.r.t. input X
    println("    $name: testing input")
    loss_test = function (X; with_grad=false)
        return loss(L, nothing, X, Cond; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(X); with_grad=true)

    if isnothing(do_flux)
        println("           $name input: Starting Flux forward")
        f_f, back, ΔX_f, do_flux = try 
            f_f, back = Flux.pullback(loss_test, deepcopy(X))
            println("           $name input: Starting Flux backward")
            ΔX_f = back(1f0)[1]
            println("           $name input: Done Flux")
            f_f, back, ΔX_f, true
        catch
            nothing, nothing, nothing, false
        end
    elseif do_flux
        println("           $name input: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(X))
        println("           $name input: Starting Flux backward")
        ΔX_f = back(1f0)[1]
        println("           $name input: Done Flux")
    end

    if do_flux
        @test f_f ≈ f0
        @test norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f)) < 2f-6
    end


    println("    $name input: First with our gradient")
    grad_test(loss_test, deepcopy(X), deepcopy(dX), deepcopy(ΔX); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT)

    if do_flux
        println("    $name input: Then with Flux's gradient")
        grad_test(loss_test, deepcopy(X), deepcopy(dX), deepcopy(ΔX_f); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT)
    end

    # Gradient test w.r.t. parameters
    clear_grad!(L)
    clear_grad!(P)
    loss_test = function (P; with_grad=false)
        return loss(L, P, X, Cond; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(P); with_grad=true)
    ΔP = deepcopy(get_grads(L))

    if do_flux
        println("           $name parameters: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(P))

        println("           $name parameters: Starting Flux backward")
        ΔP_fT = back(1f0)[1]
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        println("           $name parameters: Done Flux")


        (Y, logdet), back_Y_logdet = Flux.pullback(P -> (set_params!(L, P); forward(X, Cond, L)), deepcopy(P))
        f1, back_f1 = Flux.pullback(log_likelihood, Y)
        f, back_f = Flux.pullback((a,b) -> a - b, f1, logdet)

        Δf = 1f0
        Δf1, Δlogdet = back_f(Δf)
        ΔY, = back_f1(Δf1)
        ΔP1t, = back_Y_logdet((ΔY, Δlogdet))
        ΔP1 = [Parameter(a.data, a.grad) for a in ΔP1t]

        @test f_f ≈ f0
        @test norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f)) < 2f-6
    end

    set_params!(L, deepcopy(P))

    # Test each parameter.
    do_taylor_test = length(P) < 6
    for (i, (p, dp, Δp)) in enumerate(zip(P, dP, ΔP))
        println("    $name: testing parameter $i")
        loss_test = function (p_vec)
            P = deepcopy(P)
            P[i].data = p_vec
            return loss(L, P, X, Cond; with_grad=false)
        end

        if do_flux
            Δp_f = ΔP_f[i]
        end

        if do_taylor_test
            println("    $name parameter $i: First with our gradient")
            do_flux && @show norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f))
            grad_test(loss_test, deepcopy(p.data), deepcopy(dp.data), deepcopy(Δp.data); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
        end

        if do_flux
            do_taylor_test && @show norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f))
            @test norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f)) < 2f-6 skip=false
            do_taylor_test && println("    $name parameter $i: Then with Flux's gradient")
            do_taylor_test && grad_test(loss_test, deepcopy(p.data), deepcopy(dp.data), deepcopy(Δp_f.data); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
        end
    end

    # Test all parameters.

    # Gradient test w.r.t. parameters
    println("   $name: testing all parameters")
    clear_grad!(L)
    clear_grad!(P)
    loss_test = function (P; with_grad=false)
        return loss(L, P, X, Cond; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(P); with_grad=true)
    ΔP = deepcopy(get_grads(L))

    if do_flux
        println("           $name parameters: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(P))

        println("           $name parameters: Starting Flux backward")
        ΔP_fT = back(1f0)[1]
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        println("           $name parameters: Done Flux")

        @test norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f)) < 2f-6
    end

    println("    $name all parameter: First with our gradient")
    grad_test(loss_test, deepcopy(P), deepcopy(dP), deepcopy(ΔP); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)

    if do_flux
        @show norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f))
        println("    $name all parameter: Then with Flux's gradient")
        grad_test(loss_test, deepcopy(P), deepcopy(dP), deepcopy(ΔP_f); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
    end
end

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

name = "CorrelationScaleLayer"
@testset verbose = true "$name" begin
    println("Testing $name")

    Sm = InvertibleNetworks.CorrelationScaleLayer.forward(X)
    ΔSm = randn(TT, size(Sm))
    ΔX = apply_backward(InvertibleNetworks.CorrelationScaleLayer, ΔSm, X, Sm)

    Sm2, back = Flux.pullback(InvertibleNetworks.CorrelationScaleLayer.forward, deepcopy(X))
    @test norm(Sm - Sm2) == 0
    ΔX2, = back(ΔSm)
    @test norm(ΔX - ΔX2) < 2f-5
end


name = "CorrelationShiftLayer"
@testset verbose = true "$name" begin
    println("Testing $name")

    Tm = InvertibleNetworks.CorrelationShiftLayer.forward(X, dX)
    ΔTm = randn(TT, size(Tm))
    ΔX, ΔdX = InvertibleNetworks.CorrelationShiftLayer.backward(ΔTm, Tm, X, dX)

    Tm2, back = Flux.pullback(InvertibleNetworks.CorrelationShiftLayer.forward, deepcopy(X), deepcopy(dX))
    @test norm(Tm - Tm2) == 0
    ΔX2, ΔdX2 = back(ΔTm)
    @test norm(ΔX - ΔX2) < 2f-5
    @test norm(ΔdX - ΔdX2) < 2f-5
end

@testset verbose = true "ConditionalLayerCorrelation (n_channel_cond=$n_channel_cond)" for n_channel_cond in [n_channel, n_channel+2]
    Cond = randn(TT, nx, ny, n_channel_cond, batchsize)

    # Test with the simplest configuration.
    name = "ConditionalLayerCorrelation with no subnetworks"
    @testset verbose = true "$name (out_chan=$out_chan)" for out_chan in [split_num, 2*split_num]
        println("Testing $name")
        layer_conv1x1 = nothing
        layer_constant = LayerConstant(glorot_uniform(nx, ny, out_chan))
        L = ConditionalLayerCorrelation(nothing, layer_constant; logdet=true)

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
