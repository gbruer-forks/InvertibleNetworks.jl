export ConditionalLayerCorrelation

using Flux: get_device

struct ConditionalLayerCorrelation <: NeuralNetLayer
    prenetwork::Union{Nothing, Conv1x1, Conv1x1NoMutate}
    subnetwork::Union{ResidualBlock, LayerConstant}
    C_weights::Parameter
    scale_activation::ActivationFunction
    shift_activation::ActivationFunction
    shift_cond_scalar::Bool
    logdet::Bool
end

@Flux.functor ConditionalLayerCorrelation

# Constructor from 1x1 convolution and residual block
function ConditionalLayerCorrelation(prenetwork, subnetwork::ResidualBlock; logdet=false, scale_activation = DampedCoshLayer(), shift_activation=DampedSinhLayer(), shift_cond_scalar=true)
    subnetwork.fan == false && throw("Set ResidualBlock.fan == true")
    return ConditionalLayerCorrelation(prenetwork, subnetwork, logdet; scale_activation, shift_activation, shift_cond_scalar)
end

function ConditionalLayerCorrelation(prenetwork, subnetwork, logdet; scale_activation = DampedCoshLayer(), shift_activation=DampedSinhLayer(), shift_cond_scalar=true)
    C_weights = Parameter(nothing)
    return ConditionalLayerCorrelation(prenetwork, subnetwork, C_weights, scale_activation, shift_activation, shift_cond_scalar, logdet)
end

function ConditionalLayerCorrelation(prenetwork, subnetwork; logdet=false, scale_activation = DampedCoshLayer(), shift_activation=DampedSinhLayer(), shift_cond_scalar=true)
    return ConditionalLayerCorrelation(prenetwork, subnetwork, logdet; scale_activation, shift_activation, shift_cond_scalar)
end

function ConditionalLayerCorrelation_splitdims(n_in)
    split_num = Int(round(n_in/2))
    if split_num == 0
        split_num = 1
    end
    in_split = n_in - split_num
    return in_split, split_num
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerCorrelation) where {T,N}
    if !isnothing(L.prenetwork)
        X0, logdet = forward(X, L.prenetwork)
    else
        X0 = X
        logdet = T(0)
    end

    X1, X2 = tensor_split(X0)
    if length(X1) == 0
        X1, X2 = X2, X1
    end

    Y2 = copy(X2)

    # Cat conditioning variable C into network input
    C_X2 = tensor_cat(X2, C)
    w = forward(C_X2, L.subnetwork)

    # Split subnetwork output to get scale and shift parts.
    if size(w)[1:N-1] == size(X1)[1:N-1]
        w1 = w
        w2 = w
    else
        w1, w2 = tensor_split(w)
    end

    # Apply correlation decoupling.

    # w1 can easily be too large, such that Sm is Inf. Need to initialize it properly or limit it to a valid range.
    Sm = L.scale_activation.forward(w1)
    Tm = -L.shift_activation.forward(w2)
    if L.shift_cond_scalar
        # Get condition to use for shift. Need to be able to multiply it component-wise with X.
        Nb = size(C, N)
        if isnothing(L.C_weights.data)
            nc = prod(size(C_X2)[1:(N-1)])
            if nc == 1
                L.C_weights.data = ones(T, 1)
            else
                L.C_weights.data = glorot_uniform(nc)
            end
            L.C_weights.data = reshape(L.C_weights.data, 1, size(L.C_weights.data)...) |> get_device(C)
        end
        C_X2_scalar = L.C_weights.data * reshape(C_X2, :, Nb)
        C_X2_scalar_broadcast = reshape(C_X2_scalar, ones(Int, N-2)..., :, Nb)

        Tm = Tm .* C_X2_scalar_broadcast
    end

    Y1 = Sm .* X1 + Tm

    Y = tensor_cat(Y1, Y2)

    L.logdet == true ? (return Y, logdet + scale_logdet_forward(Sm)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerCorrelation; save=false) where {T,N}

    Y1, Y2 = tensor_split(Y)
    if length(Y1) == 0
        Y1, Y2 = Y2, Y1
    end

    X2 = copy(Y2)

    # Cat conditioning variable C into network input.
    C_X2 = tensor_cat(X2, C)
    w = forward(C_X2, L.subnetwork)

    # Split subnetwork output to get scale and shift parts.
    if size(w)[1:N-1] == size(Y1)[1:N-1]
        w1 = w
        w2 = w
    else
        w1, w2 = tensor_split(w)
    end

    # Invert correlation decoupling.
    Sm = L.scale_activation.forward(w1)
    Tm = -L.shift_activation.forward(w2)
    if L.shift_cond_scalar
        # Get condition to use for shift. Need to be able to multiply it component-wise with X.
        Nb = size(C, N)
        C_X2_scalar = L.C_weights.data * reshape(C_X2, :, Nb)
        C_X2_scalar_broadcast = reshape(C_X2_scalar, (1 for i in 1:N-2)..., :, Nb)
        Tm = Tm .* C_X2_scalar_broadcast
    end
    X1 = (Y1 - Tm) ./ Sm

    X0 = tensor_cat(X1, X2)
    if !isnothing(L.prenetwork)
        X, logdet = inverse(X0, L.prenetwork)
    else
        X = X0
        logdet = T(0)
    end

    save == true ? (return X, X1, X2, w, w1, w2, Sm, Tm) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerCorrelation;) where {T,N}

    # Recompute forward state
    X, X1, X2, w, w1, w2, Sm, Tm = inverse(Y, C, L; save=true)

    # Backpropagate coupling.
    ΔY1, ΔY2 = tensor_split(ΔY)
    if length(ΔY1) == 0
        ΔY1, ΔY2 = ΔY2, ΔY1
    end

    ΔTm = copy(ΔY1)

    ΔSm = ΔY1 .* X1
    if L.logdet
        ΔSm -= scale_logdet_backward(Sm)
    end
    ΔX1 = ΔY1 .* Sm

    C_X2 = tensor_cat(X2, C)

    # Backpropagate activations.
    if L.shift_cond_scalar
        # Get condition to use for shift. Need to be able to multiply it component-wise with X.
        Nb = size(C, N)
        C_X2_scalar = L.C_weights.data * reshape(C_X2, :, Nb)
        C_X2_scalar_broadcast = reshape(C_X2_scalar, (1 for i in 1:N-2)..., :, Nb)

        Tm = Tm ./ C_X2_scalar_broadcast
        ΔC_X2_scalar_broadcast = Tm .* ΔTm

        ΔC_X2_scalar_broadcast = reshape(ΔC_X2_scalar_broadcast, :, size(C_X2_scalar)[2:end]...)
        ΔC_X2_scalar = sum(ΔC_X2_scalar_broadcast; dims=1)

        ΔC_X2_vector = L.C_weights.data' * ΔC_X2_scalar
        ΔC_weights = ΔC_X2_scalar * reshape(C_X2, :, Nb)'

        nc = prod(size(C)[1:(N-1)])
        if nc != 1
            isnothing(L.C_weights.grad) ? (L.C_weights.grad = ΔC_weights) : (L.C_weights.grad += ΔC_weights)
        end

    else
        ΔC_X2_scalar_broadcast = 0
    end
    Δw1 = apply_backward(L.scale_activation, ΔSm, w1, Sm)
    Δw2 = apply_backward(L.shift_activation, -ΔTm, w2, -Tm)
    if L.shift_cond_scalar
        Δw2 = Δw2 .* C_X2_scalar_broadcast
    end

    # Join scale and shift parts.
    if size(w)[1:N-1] == size(X1)[1:N-1]
        Δw = Δw1 .+ Δw2
    else
        Δw = tensor_cat(Δw1, Δw2)
    end

    # Backpropagate subnetwork.
    ΔX2_ΔC = backward(Δw, C_X2, L.subnetwork)
    ΔX2, ΔC = tensor_split(ΔX2_ΔC; split_index=size(ΔY2)[N-1])
    ΔX2 += ΔY2
    if L.shift_cond_scalar
        ΔX2_vector, ΔC_vector = tensor_split(reshape(ΔC_X2_vector, size(C_X2)); split_index=size(ΔY2)[N-1])

        ΔX2 += reshape(ΔX2_vector, size(ΔX2))
        ΔC += reshape(ΔC_vector, size(ΔC))
    end

    # Backpropagate prenetwork.

    ΔX0 = tensor_cat(ΔX1, ΔX2)
    if !isnothing(L.prenetwork)
        ΔX = inverse((ΔX0, tensor_cat(X1, X2)), L.prenetwork)[1]
    else
        ΔX = ΔX0
    end

    return ΔX, X, ΔC
end

scale_logdet_forward(S) = sum(log.(abs.(S))) / size(S)[end]
scale_logdet_backward(S) = 1f0./ S / size(S)[end]
