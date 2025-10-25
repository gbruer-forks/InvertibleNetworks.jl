export ConditionalCouplingLayer

using Flux: get_device

struct ConditionalCouplingLayer <: NeuralNetLayer
    prenetwork::Union{Nothing, Conv1x1, Conv1x1NoMutate}
    subnetwork::Union{ResidualBlock, LayerConstant, LayerStack}
    invertible_operator::Union{AffineCouplingOperator, RQSpline1Operator, ConditionalDecorrelationOperator}
    logdet::Bool
end

@Flux.functor ConditionalCouplingLayer

function ConditionalCouplingLayer(prenetwork, subnetwork::ResidualBlock; logdet=false, kwargs...)
    subnetwork.fan == false && throw("Set ResidualBlock.fan == true")
    return ConditionalCouplingLayer(prenetwork, subnetwork, logdet; kwargs...)
end

function ConditionalCouplingLayer(prenetwork, subnetwork; logdet=false, kwargs...)
    return ConditionalCouplingLayer(prenetwork, subnetwork, logdet; kwargs...)
end

function ConditionalCouplingLayer(prenetwork, subnetwork, invertible_operator; logdet=true)
    return ConditionalCouplingLayer(prenetwork, subnetwork, invertible_operator, logdet)
end

# For backwards compatibility.
function ConditionalCouplingLayer(prenetwork, subnetwork, logdet::Bool; scale_activation = DampedCoshLayer(), shift_activation=DampedSinhLayer(), shift_cond_scalar=true)
    invertible_operator = AffineCouplingOperator(; shift_cond_scalar, scale_activation, shift_activation)
    return ConditionalCouplingLayer(prenetwork, subnetwork, invertible_operator, logdet)
end

function ConditionalCouplingLayer_splitdims(n_in)
    split_num = Int(round(n_in/2))
    if split_num == 0
        split_num = 1
    end
    in_split = n_in - split_num
    return in_split, split_num
end

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalCouplingLayer) where {T,N}
    if !isnothing(L.prenetwork)
        X0, logdet_pre = forward(X, L.prenetwork)
    else
        X0 = X
        logdet_pre = T(0)
    end

    X1, X2 = tensor_split(X0)
    if length(X1) == 0
        X1, X2 = X2, X1
    end

    Y2 = copy(X2)

    # Cat conditioning variable C into network input
    C_X2 = tensor_cat(X2, C)
    w = forward(C_X2, L.subnetwork)

    # Invertible operator uses w to invertibly transform X1.
    Y1, logdet = forward(X1, C_X2, w, L.invertible_operator)

    Y = tensor_cat(Y1, Y2)

    L.logdet == true ? (return Y, logdet_pre + logdet) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalCouplingLayer; save=false) where {T,N}

    Y1, Y2 = tensor_split(Y)
    if length(Y1) == 0
        Y1, Y2 = Y2, Y1
    end

    X2 = copy(Y2)

    # Cat conditioning variable C into network input.
    C_X2 = tensor_cat(X2, C)
    w = forward(C_X2, L.subnetwork)

    # Invertible operator uses w to invertibly get X1.
    X1, saved = inverse(Y1, C_X2, w, L.invertible_operator)

    X0 = tensor_cat(X1, X2)
    if !isnothing(L.prenetwork)
        X, logdet = inverse(X0, L.prenetwork)
    else
        X = X0
        logdet = T(0)
    end

    save == true ? (return X, X1, X2, w, saved) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalCouplingLayer) where {T,N}
    # Recompute forward state
    X, X1, X2, w, saved = inverse(Y, C, L; save=true)

    # Backpropagate coupling.
    ΔY1, ΔY2 = tensor_split(ΔY)
    if length(ΔY1) == 0
        ΔY1, ΔY2 = ΔY2, ΔY1
    end
    C_X2 = tensor_cat(X2, C)

    # Invertible operator applies adjoint Jacobian of Y1.

    # Δlogdet = L.logdet ? T(-1) / size(Y)[end] : T(0)
    # ΔX1, ΔC_X2_invop, Δw = backward(ΔY1, Δlogdet, X1, C_X2, w, L.invertible_operator, saved)
    ΔX1, ΔC_X2_invop, Δw = backward(ΔY1, X1, C_X2, w, L.invertible_operator, saved)

    # Backpropagate subnetwork.
    ΔC_X2_subnet = backward(Δw, C_X2, L.subnetwork)
    # @show size(X) size(X1) size(X2) size(C) size(C_X2) size(ΔC_X2_invop) size(ΔC_X2_subnet)
    ΔC_X2 = ΔC_X2_invop + ΔC_X2_subnet

    ΔX2, ΔC = tensor_split(ΔC_X2; split_index=size(ΔY2)[N-1])
    ΔX2 = ΔX2 + ΔY2

    # Backpropagate prenetwork.

    ΔX0 = tensor_cat(ΔX1, ΔX2)
    if !isnothing(L.prenetwork)
        ΔX = inverse((ΔX0, tensor_cat(X1, X2)), L.prenetwork)[1]
    else
        ΔX = ΔX0
    end

    return ΔX, X, ΔC
end

