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

@testset verbose = true "ConditionalLayerGlow (n_channel_cond=$n_channel_cond)" for n_channel_cond in [n_channel, n_channel+2]
    println("Testing ConditionalLayerGlow")
    Cond = randn(TT, nx, ny, n_channel_cond, batchsize)
    out_chan = 2*split_num
    k1 = 1
    k2 = 1
    p1 = 0
    p2 = 0
    fan = true
    n_hidden = 4
    layer_conv1x1 = Conv1x1NoMutate(n_channel; logdet=false)
    activation = SigmoidLayer()
    final_activation = SigmoidLayer()
    layer_resblock = ResidualBlock(in_split+n_channel_cond, n_hidden; n_out=out_chan, k1, k2, p1, p2, fan, activation, final_activation)
    L = ConditionalLayerGlow(layer_conv1x1, layer_resblock; logdet=true)

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
