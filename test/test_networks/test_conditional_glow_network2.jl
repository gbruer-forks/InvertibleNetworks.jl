using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse

include("../conditional_test.jl")

# Random seed
Random.seed!(11)


# Input
in_shape = (3, 11, 5)
cond_shape = (3, 11, 7)
batchsize = 13
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

n_in = in_shape[end]
n_cond = cond_shape[end]

name = "NetworkConditionalGlow"
@testset verbose = true "$name (K=$K)" for K in [1, 3]
    L = 1
    println("Testing $name")
    n_hidden = 3
    activation = SigmoidLayer()
    rb_activation = SigmoidLayer()
    kwargs = (; k1=3, k2=1, p1=0, p2=0, s1=1, s2=1, ndims=2, activation, rb_activation)
    G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; kwargs...)
    P = get_params_as_type(G, X, Cond, TT)

    G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; kwargs...)
    P0 = get_params_as_type(G0, X0, Cond0, TT)
    dP = P0 - P

    # conditional_layer_test_inverse(G, X, Cond, dX; returns_CondY=true)
    conditional_layer_test_gradient(G, P, dP, X, Cond, dX; name, returns_CondY=true)
end
