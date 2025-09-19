using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux

using InvertibleNetworks: forward, backward, inverse

include("../noninvertible_test.jl")

# Random seed
Random.seed!(11)


# Input
in_shape = (3, 11, 5)
cond_shape = (3, 11, 7)
batchsize = 7
in_split, split_num = InvertibleNetworks.ConditionalLayerCorrelation_splitdims(in_shape[end])

# Input images
TT = Float64
X = randn(TT, in_shape..., batchsize)
X0 = randn(TT, in_shape..., batchsize)
Y0 = randn(TT, in_shape..., batchsize)
dX = X - X0

function get_params_as_type(G, X, TT)
    if TT != Float32
        forward(Float32.(X), G)
        P = deepcopy(get_params(G))
        for p in P
            p.data = TT.(p.data)
        end
        set_params!(G, deepcopy(P))
    else
        forward(X, G)
        P = deepcopy(get_params(G))
    end
    return P
end

n_in = in_shape[end]
n_cond = cond_shape[end]

name = "ResidualBlock"
@testset verbose = true "$name" begin
# if true
    println("Testing $name")
    n_hidden = 5
    activation = SigmoidLayer()
    final_activation = SoftplusLayer()
    kwargs = (; k1=2, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=2, activation, final_activation)
    G = ResidualBlock(n_in, n_hidden; kwargs...)
    P = get_params_as_type(G, X, TT)

    G0 = ResidualBlock(n_in, n_hidden; kwargs...)
    G0.b1.data = randn(eltype(G.b1.data), size(G.b1.data)...)
    G0.b2.data = randn(eltype(G.b1.data), size(G.b2.data)...)
    P0 = get_params_as_type(G0, X0, TT)
    dP = P0 - P

    noninvertible_layer_test_gradient(G, P, dP, X, dX; name)
end
