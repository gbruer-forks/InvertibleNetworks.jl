using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse

include("../grad_test.jl")

# Random seed
Random.seed!(11)

###################################################################################################
# Test invertibility

function conditional_network_test_inverse(G, X, Cond, dX)
    Y, YC, logdet = forward(X, Cond, G)
    X_ = inverse(Y, YC, G)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-5)

    X_ = forward(inverse(X, YC, G), Cond, G)[1]
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-5)
end

function conditional_network_test_gradient(G, P, dP, X, Cond, dX; name, do_flux=nothing)
    # Loss Function
    loss = function (G, P, X, Cond; with_grad)
        if !isnothing(P)
            set_params!(G, P)
        end
        Y, YC, logdet = forward(X, Cond, G)
        f = log_likelihood(Y) - logdet
        if with_grad
            ΔY = ∇log_likelihood(Y)
            ΔX = backward(ΔY, Y, YC, G)[1]
            return f, ΔX
        end
        return f
    end

    # Gradient test w.r.t. input X
    println("    $name: testing input")
    loss_test = function (X; with_grad=false)
        return loss(G, nothing, X, Cond; with_grad)
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
        @show norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f))
        @test f_f ≈ f0
        @test norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f)) < 2f-6
    end

    println("    $name input: First with our gradient")
    grad_test(loss_test, deepcopy(X), deepcopy(dX), deepcopy(ΔX); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT)

    if do_flux
        @show norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f))
        println("    $name input: Then with Flux's gradient")
        grad_test(loss_test, deepcopy(X), deepcopy(dX), deepcopy(ΔX_f); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT)
        @test f_f ≈ f0
        @test norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f)) < 2f-6
    end

    # Gradient test w.r.t. parameters
    clear_grad!(G)
    clear_grad!(P)
    loss_test = function (P; with_grad=false)
        return loss(G, P, X, Cond; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(P); with_grad=true)
    ΔP = deepcopy(get_grads(G))

    if do_flux
        println("           $name parameters: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(P))

        println("           $name parameters: Starting Flux backward")
        ΔP_fT = back(1f0)[1]
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        println("           $name parameters: Done Flux")

        (Y, YC, logdet), back_Y_YC_logdet = Flux.pullback(P -> (set_params!(G, P); forward(X, Cond, G)), deepcopy(P))
        f1, back_f1 = Flux.pullback(log_likelihood, Y)
        f, back_f = Flux.pullback((a,b) -> a - b, f1, logdet)

        Δf = 1f0
        # @show Δf
        Δf1, Δlogdet = back_f(Δf)
        # @show Δf1
        # @show Δlogdet
        ΔY, = back_f1(Δf1)
        # @show ΔY
        ΔYC = nothing
        ΔP1t, = back_Y_YC_logdet((ΔY, ΔYC, Δlogdet))
        ΔP1 = [Parameter(a.data, a.grad) for a in ΔP1t]

        @test f_f ≈ f0
        @test norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f)) < 2f-6
    end

    set_params!(G, deepcopy(P))

    # Test each parameter.
    do_taylor_test = length(P) < 6
    if do_taylor_test || do_flux
        for (i, (p, dp, Δp)) in enumerate(zip(P, dP, ΔP))
            println("    $name: testing parameter $i")
            loss_test = function (p_vec)
                P = deepcopy(P)
                P[i].data = p_vec
                return loss(G, P, X, Cond; with_grad=false)
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
    end

    # Test all parameters.

    # Gradient test w.r.t. parameters
    println("   $name: testing all parameters")
    clear_grad!(G)
    clear_grad!(P)
    loss_test = function (P; with_grad=false)
        return loss(G, P, X, Cond; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(P); with_grad=true)
    ΔP = deepcopy(get_grads(G))
    # @show P ΔP

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
    if do_flux
        @show norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f))
    end
    grad_test(loss_test, deepcopy(P), deepcopy(dP), deepcopy(ΔP); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)

    if do_flux
        @show norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f))
        println("    $name all parameter: Then with Flux's gradient")
        grad_test(loss_test, deepcopy(P), deepcopy(dP), deepcopy(ΔP_f); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
    end
end

# Input
in_shape = (3, 11, 5)
cond_shape = (3, 11, 7)
batchsize = 7
in_split, split_num = InvertibleNetworks.ConditionalLayerCorrelation_splitdims(in_shape[end])

# Input images
TT = Float64
X = randn(TT, in_shape..., batchsize)
Cond = randn(TT, cond_shape..., batchsize)
Cond0 = randn(TT, cond_shape..., batchsize)
X0 = randn(TT, in_shape..., batchsize)
Y0 = randn(TT, in_shape..., batchsize)
dX = X - X0

function get_params_as_type(G, X, Cond, TT)
    if TT != Float32
        forward(Float32.(X), Float32.(Cond), G)
        P = deepcopy(get_params(G))
        for p in P
            p.data = TT.(p.data)
        end
        set_params!(G, deepcopy(P))
    else
        forward(X, Cond, G)
        P = deepcopy(get_params(G))
    end
    return P
end

# Test with the simplest configuration.
name = "(subnetworks = ConstantLayer)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    println("Testing $name")
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K)
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K)
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end

# Test with Conv1x1NoMutate.
name = "(subnetworks = ConstantLayer; prenetworks = Conv1x1NoMutate)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end

# Test with Conv1x1.
name = "(subnetworks = ConstantLayer; prenetworks = Conv1x1)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    prenetwork_generator = in_shape -> Conv1x1(in_shape[end]; logdet=true)
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end

# Test with ResBlock.
name = "(subnetworks = ResBlock)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    subnetwork_generator = function (in_shape, out_shape)
        activation = SigmoidLayer()
        final_activation = IdentityActivation()
        k1 = 3
        k2 = 1
        p1 = 1
        p2 = 0
        s1 = 1
        s2 = 1
        n_hidden = 2
        ndims = 2
        ResidualBlock(in_shape[end], n_hidden; n_out=2*out_shape[end], activation, k1, k2, p1, p2, s1, s2, fan=true, ndims, final_activation)
    end
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        subnetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        subnetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end


# Test with ResBlock and Conv1x1.
name = "(subnetworks = ResBlock; prenetworks = Conv1x1NoMutate)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    subnetwork_generator = function (in_shape, out_shape)
        activation = SigmoidLayer()
        final_activation = IdentityActivation()
        kwargs = (; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, activation, final_activation)
        n_hidden = 4
        ResidualBlock(in_shape[end], n_hidden; n_out=2*out_shape[end], kwargs..., fan=true)
    end
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        prenetwork_generator,
        subnetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        prenetwork_generator,
        subnetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end

# Test with ActNorm.
name = "(state_network=ActNorm, cond_network=ActNorm)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    cond_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=false)
    state_initial_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_middle_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_final_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end


# Test with ResidualBlock and ActNorm.
name = "(state_network=ActNorm, cond_network=ActNorm, subnetwork=ResidualBlock)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    subnetwork_generator = function (in_shape, out_shape)
        activation = SigmoidLayer()
        final_activation = IdentityActivation()
        ndims = length(in_shape) - 1
        kwargs = (; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims, activation, final_activation)
        n_hidden = 4
        ResidualBlock(in_shape[end], n_hidden; n_out=2*out_shape[end], kwargs..., fan=true)
    end
    cond_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=false)
    state_initial_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_middle_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_final_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end

# Test with ResidualBlock, ActNorm, and Conv1x1.
name = "(state_network=ActNorm; cond_network=ActNorm; prenetwork=Conv1x1; subnetwork=ResidualBlock)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    subnetwork_generator = function (in_shape, out_shape)
        activation = SigmoidLayer()
        final_activation = IdentityActivation()
        ndims = length(in_shape) - 1
        kwargs = (; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims, activation, final_activation)
        n_hidden = 4
        ResidualBlock(in_shape[end], n_hidden; n_out=2*out_shape[end], kwargs..., fan=true)
    end
    cond_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=false)
    state_initial_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_middle_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_final_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_network_test_inverse(G, X, Cond, dX)
    conditional_network_test_gradient(G, P, dP, X, Cond, dX; name)
end
