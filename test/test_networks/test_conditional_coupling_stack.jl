using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse

include("../grad_test.jl")
include("../conditional_test.jl")

# Random seed
Random.seed!(11)

###################################################################################################
# Input
in_shape = (3, 11, 5)
cond_shape = (3, 11, 7)
batchsize = 7
in_split, split_num = InvertibleNetworks.ConditionalCouplingLayer_splitdims(in_shape[end])

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
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K)
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K)
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
end

# Test with Conv1x1NoMutate.
name = "(subnetworks = ConstantLayer; prenetworks = Conv1x1NoMutate)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
end

# Test with Conv1x1.
name = "(subnetworks = ConstantLayer; prenetworks = Conv1x1)"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    out_chan = split_num
    println("Testing $name")
    prenetwork_generator = in_shape -> Conv1x1(in_shape[end]; logdet=true)
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        prenetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
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
        ResidualBlock(in_shape[end], n_hidden; n_out=out_shape[end], activation, k1, k2, p1, p2, s1, s2, fan=true, ndims, final_activation)
    end
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        subnetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        subnetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
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
        ResidualBlock(in_shape[end], n_hidden; n_out=out_shape[end], kwargs..., fan=true)
    end
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        prenetwork_generator,
        subnetwork_generator,
    )

    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        prenetwork_generator,
        subnetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
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
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
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
        ResidualBlock(in_shape[end], n_hidden; n_out=out_shape[end], kwargs..., fan=true)
    end
    cond_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=false)
    state_initial_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_middle_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_final_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
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
        ResidualBlock(in_shape[end], n_hidden; n_out=out_shape[end], kwargs..., fan=true)
    end
    cond_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=false)
    state_initial_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_middle_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_final_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
end


# Test with ResidualBlock, ActNorm, and Conv1x1.
name = "(state_network=ActNorm; cond_network=ActNorm; prenetwork=Conv1x1; subnetwork=ResidualBlock; split=false)"
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
        ResidualBlock(in_shape[end], n_hidden; n_out=out_shape[end], kwargs..., fan=true)
    end
    cond_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=false)
    state_initial_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_middle_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    state_final_network_generator = in_shape -> ActNorm(in_shape[end]; logdet=true)
    prenetwork_generator = in_shape -> Conv1x1NoMutate(in_shape[end]; logdet=true)
    G = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
        coupling_layer_params = (; split=false),
    )
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalCouplingStack(in_shape, cond_shape, L, K;
        cond_network_generator,
        state_initial_network_generator,
        state_middle_network_generator,
        state_final_network_generator,
        subnetwork_generator,
        prenetwork_generator,
        coupling_layer_params = (; split=false),
    )
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
end
