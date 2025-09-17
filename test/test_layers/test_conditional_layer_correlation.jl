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
        # @show f0 f_f f
        # @show norm(ΔX) norm(ΔX_f)
        # @test f ≈ f0
        # @test norm(ΔX) ≈ norm(ΔX_f)
        # @show norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f))
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
        # @show Δf
        Δf1, Δlogdet = back_f(Δf)
        # @show Δf1
        # @show Δlogdet
        ΔY, = back_f1(Δf1)
        # @show ΔY
        ΔP1t, = back_Y_logdet((ΔY, Δlogdet))
        ΔP1 = [Parameter(a.data, a.grad) for a in ΔP1t]


        # @show f0 f_f
        # @show typeof(ΔP) typeof(ΔP_f) typeof(ΔP1)
        # @show norm(ΔP) norm(ΔP_f)
        # @show norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f))
        # @show norm(ΔP[1] - ΔP_f[1]) ./ max(norm(ΔP[1]), norm(ΔP_f[1]))

        @test f_f ≈ f0
        @test norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f)) < 2f-6
    end

    set_params!(L, deepcopy(P))
    # println("Really breaking it down now")
    # let X=X, C=Cond, L=L, N=4
    #     first = function (X, C, L)
    #         X0 = X

    #         X1, X2 = InvertibleNetworks.tensor_split(X0)
    #         if length(X1) == 0
    #             X1, X2 = X2, X1
    #         end

    #         Y2 = copy(X2)

    #         # Cat conditioning variable C into network input
    #         w = forward(InvertibleNetworks.tensor_cat(X2, C), L.subnetwork)
    #         return X1, Y2, w
    #     end
    #     (X1, Y2, w), back_X1_Y2_w = Flux.pullback(first, X, C, L)

    #     second = function (w, X1, L, C)
    #         # Split subnetwork output to get scale and shift parts.
    #         if size(w)[1:N-1] == size(X1)[1:N-1]
    #             w1 = w
    #             w2 = w
    #         else
    #             w1, w2 = InvertibleNetworks.tensor_split(w)
    #         end

    #         # Get condition to use for shift.
    #         Nb = 1
    #         C_scalar = L.C_weights.data * reshape(C, :, Nb)
    #         C_scalar = reshape(C_scalar, ones(Int, N-2)..., :, Nb)
    #         return w1, w2, C_scalar
    #     end
    #     (w1, w2, C_scalar), back_w1_w2_C_scalar = Flux.pullback(second, w, X1, L, C)

    #     third = function (w1, w2, C_scalar)
    #         # Apply correlation decoupling.
    #         Sm = InvertibleNetworks.CorrelationScaleLayer.forward(w1)
    #         Tm = InvertibleNetworks.CorrelationShiftLayer.forward(w2, C_scalar)
    #         return Sm, Tm
    #     end
    #     (Sm, Tm), back_Sm_Tm = Flux.pullback(third, w1, w2, C_scalar)

    #     fourth = function (Sm, X1, Tm, Y2)
    #         Y1 = Sm .* X1 + Tm
    #         Y = tensor_cat(Y1, Y2)
    #         return Y
    #     end
    #     Y, back_Y = Flux.pullback(fourth, Sm, X1, Tm, Y2)

    #     logdet, back_logdet = Flux.pullback(InvertibleNetworks.scale_logdet_forward, Sm)

    #     # Go backwards.
    #     Δx_Sm_1, = back_logdet(Δlogdet)
    #     Δx_Sm_2, Δx_X1a, Δx_Tm, Δx_Y2 = back_Y(ΔY)
    #     Δx_Sm, = Δx_Sm_1 + Δx_Sm_2
    #     Δx_w1, Δx_w2, Δx_C_scalar = back_Sm_Tm((Δx_Sm, Δx_Tm))
    #     Δx_w, Δx_X1b, Δx_L1, Δx_C1 = back_w1_w2_C_scalar((Δx_w1, Δx_w2, Δx_C_scalar))
    #     Δx_X, Δx_C2, Δx_L2 = back_X1_Y2_w((Δx_X1a, Δx_Y2, Δx_w))
    #     # @show C_scalar w2 Tm
    # end

    # error("DONE")

    # Test each parameter.
    # @show P ΔP ΔP_f
    for (i, (p, dp, Δp)) in enumerate(zip(P, dP, ΔP))
        println("    $name: testing parameter $i")
        loss_test = function (p_vec)
            P = deepcopy(P)
            P[i].data = p_vec
            return loss(L, P, X, Cond; with_grad=false)
        end

        println("    $name parameter $i: First with our gradient")
        grad_test(loss_test, deepcopy(p.data), deepcopy(dp.data), deepcopy(Δp.data); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
        if do_flux
            Δp_f = ΔP_f[i]
            @show norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f))
            @test norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f)) < 2f-6 skip=false
            println("    $name parameter $i: Then with Flux's gradient")
            grad_test(loss_test, deepcopy(p.data), deepcopy(dp.data), deepcopy(Δp_f.data); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
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
    # @show P ΔP

    if do_flux
        println("           $name parameters: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(P))

        println("           $name parameters: Starting Flux backward")
        ΔP_fT = back(1f0)[1]
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        println("           $name parameters: Done Flux")
        # @show P ΔP_f

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
batchsize = 7
in_split, split_num = InvertibleNetworks.ConditionalLayerCorrelation_splitdims(n_channel)

# Input images
TT = Float64
X = randn(TT, nx, ny, n_channel, batchsize)
Cond = randn(TT, nx, ny, n_channel, batchsize)
X0 = randn(TT, nx, ny, n_channel, batchsize)
Y0 = randn(TT, nx, ny, n_channel, batchsize)
dX = X - X0

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
    layer_conv1x1 = Conv1x1(n_channel)
    layer_conv1x1_nomutate = Conv1x1NoMutate(n_channel)

    P = deepcopy(get_params(layer_conv1x1))
    if TT != Float32
        for p in P
            p.data = TT.(p.data)
        end
        set_params!(layer_conv1x1, deepcopy(P))
    end
    set_params!(layer_conv1x1_nomutate, deepcopy(P))

    Y = forward(X, layer_conv1x1)
    Y_f = forward(X, layer_conv1x1_nomutate)

    # @show norm(Y - Y_f) ./ norm(Y)
    @test norm(Y - Y_f) ./ norm(Y) < 2f-6

    ΔY = randn(eltype(Y), size(Y))
    ΔX = inverse((ΔY, Y), layer_conv1x1)[1]
    ΔP = deepcopy(get_grads(layer_conv1x1))

    forward_test = function (X, P)
        set_params!(layer_conv1x1_nomutate, P)
        forward(X, layer_conv1x1_nomutate)
    end
    Y_f, back = Flux.pullback(forward_test, deepcopy(X), deepcopy(P))

    # @show norm(Y - Y_f) ./ norm(Y)
    @test norm(Y - Y_f) ./ norm(Y) < 2f-6

    ΔX_f, ΔP_fT = back(ΔY)
    ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

    # @show norm(ΔX - ΔX_f) ./ norm(ΔX)
    # @show norm(ΔP - ΔP_f) ./ norm(ΔP)
    @test norm(ΔX - ΔX_f) ./ norm(ΔX) < 2f-6
    @test norm(ΔP - ΔP_f) ./ norm(ΔP) < 2f-6
end

# Test with Conv1x1NoMutate.
name = "ConditionalLayerCorrelation with Conv1x1NoMutate"
@testset verbose=true "$name" begin
    println("Testing $name")
    layer_conv1x1 = Conv1x1NoMutate(n_channel)
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
    layer_conv1x1 = Conv1x1(n_channel)
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
    layer_resblock = ResidualBlock(in_split+n_channel, n_hidden; n_out=out_chan, k1, k2, p1, p2, fan, activation, final_activation)
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
    layer_conv1x1 = Conv1x1NoMutate(n_channel)
    activation = SoftplusLayer()
    final_activation = IdentityActivation()
    layer_resblock = ResidualBlock(in_split+n_channel, n_hidden; n_out=out_chan, k1, k2, p1, p2, fan, activation, final_activation)
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
