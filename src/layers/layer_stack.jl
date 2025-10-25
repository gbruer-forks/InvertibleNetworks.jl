export LayerStack

using Flux: get_device

StackLayerNetworkType = Union{ResidualBlock, LayerConstant}
struct LayerStack <: NeuralNetLayer
    subnetworks::Vector{NeuralNetLayer}
    splits::Vector{Int}
end

@Flux.functor LayerStack

function LayerStack(subnetworks::Vector{NeuralNetLayer})
    splits = zeros(Int, length(subnetworks) + 1)
    return LayerStack(subnetworks, splits)
end

function LayerStack(subnetwork::ResidualBlock)
    subnetwork.fan == false && throw("Set ResidualBlock.fan == true")
    return LayerStack([subnetwork])
end

function LayerStack(subnetwork::StackLayerNetworkType)
    return LayerStack([subnetwork])
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, L::LayerStack) where {T,N}
    d = max(1, N-1)
    Ys = [forward(X, subnetwork) for subnetwork in L.subnetworks]
    Flux.ignore() do
        L.splits[1] = 1
        for (i, Yi) in enumerate(Ys)
            L.splits[i+1] = L.splits[i] + size(Yi, d)
        end
    end
    Y = cat(Ys...; dims=d)
    return Y
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, L::LayerStack) where {T,N}
    d = max(1, N-1)
    ΔX = zero(X)
    for (i, subnetwork) in enumerate(L.subnetworks)
        ΔYi = selectdim(ΔY, d, L.splits[i]:(L.splits[i+1]-1))
        ΔX += backward(ΔYi, X, L.subnetworks[i])
    end
    return ΔX
end

