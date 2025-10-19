export AffineCouplingOperator

struct AffineCouplingOperator <: NeuralNetLayer
    C_weights::Union{Parameter, Nothing}
    shift_cond_scalar::Bool
    scale_activation::ActivationFunction
    shift_activation::ActivationFunction
end

function AffineCouplingOperator(; scale_activation = DampedCoshLayer(), shift_activation=DampedSinhLayer(), shift_cond_scalar=true)
    C_weights = shift_cond_scalar ? Parameter(nothing) : nothing
    return AffineCouplingOperator(C_weights, shift_cond_scalar, scale_activation, shift_activation)
end

function forward(X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{Tw, Nw}, L::AffineCouplingOperator) where {T,Tw,N,NcNx2,Nw}
    # Split subnetwork output to get scale and shift parts.
    if size(w)[1:N-1] == size(X1)[1:N-1]
        w1 = w
        w2 = w
    else
        w1, w2 = tensor_split(w)
    end

    # Apply correlation decoupling.
    Sm = L.scale_activation.forward(w1)
    Tm_1 = L.shift_activation.forward(w2)
    if L.shift_cond_scalar
        # Get condition to use for shift. Need to be able to multiply it component-wise with X.
        Nb = size(X1, N)
        if isnothing(L.C_weights.data)
            nc = prod(size(C_X2)[1:end-1])
            if nc == 1
                L.C_weights.data = ones(T, size(C_X2)[1:end-1])
            else
                L.C_weights.data = glorot_uniform(nc)
            end
            L.C_weights.data = reshape(L.C_weights.data, 1, size(L.C_weights.data)...) |> get_device(X1)
        end
        if prod(size(C_X2)[1:(N-1)]) == 1
            C_X2_scalar = reshape(C_X2, :, Nb)
        else
            # Get condition to use for shift. Need to be able to multiply it component-wise with X.
            C_X2_scalar = L.C_weights.data * reshape(C_X2, :, Nb)
        end
        C_X2_scalar_broadcast = reshape(C_X2_scalar, ones(Int, N-2)..., :, Nb)
        Tm = -Tm_1 .* C_X2_scalar_broadcast
    else
        Tm = Tm_1
    end

    Y1 = Sm .* X1 + Tm
    logdet = scale_logdet_forward(Sm) / size(X1, N)
    return Y1, logdet
end


function inverse(Y1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::AffineCouplingOperator) where {T,N,NcNx2,Nw}
    # Split subnetwork output to get scale and shift parts.
    if size(w)[1:end-1] == size(Y1)[1:end-1]
        w1 = w
        w2 = w
    else
        w1, w2 = tensor_split(w)
    end

    # Invert correlation decoupling.
    Sm = L.scale_activation.forward(w1)
    Tm_1 = L.shift_activation.forward(w2)
    if L.shift_cond_scalar
        Nb = size(Y1, N)
        if prod(size(C_X2)[1:(NcNx2-1)]) == 1
            C_X2_scalar = reshape(C_X2, :, Nb)
        else
            # Get condition to use for shift. Need to be able to multiply it component-wise with X.
            C_X2_scalar = L.C_weights.data * reshape(C_X2, :, Nb)
        end
        C_X2_scalar_broadcast = reshape(C_X2_scalar, ones(Int, N-2)..., :, Nb)
        Tm = -Tm_1 .* C_X2_scalar_broadcast
    else
        Tm = Tm_1
    end
    X1 = (Y1 - Tm) ./ Sm
    return X1, (; w1, w2, Sm, Tm_1, Tm)
end

function backward(ΔY1::AbstractArray{T, N}, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::AffineCouplingOperator, saved) where {T,N,NcNx2,Nw}
    Δlogdet = T(-1) / size(ΔY1, N)
    return backward(ΔY1, Δlogdet, X1, C_X2, w, L, saved)
end

function backward(ΔY1::AbstractArray{T, N}, Δlogdet::T, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::AffineCouplingOperator, saved) where {T,N,NcNx2,Nw}
    (; w1, w2, Sm, Tm_1, Tm) = saved
    ΔTm = copy(ΔY1)

    ΔSm = ΔY1 .* X1
    ΔSm = ΔSm + Δlogdet * scale_logdet_backward(Sm)
    ΔX1 = ΔY1 .* Sm

    # Backpropagate activations.
    if L.shift_cond_scalar
        Nb = size(X1, N)
        C_X2_vector = reshape(C_X2, :, Nb)
        if prod(size(C_X2)[1:(N-1)]) == 1
            C_X2_scalar = C_X2_vector
        else
            # Get condition to use for shift. Need to be able to multiply it component-wise with X.
            C_X2_scalar = L.C_weights.data * C_X2_vector
        end
        C_X2_scalar_broadcast = reshape(C_X2_scalar, (1 for i in 1:N-2)..., :, Nb)

        ΔC_X2_scalar_broadcast = -ΔTm .* Tm_1
        ΔTm_1 = -ΔTm .* C_X2_scalar_broadcast

        ΔC_X2_scalar = sum(reshape(ΔC_X2_scalar_broadcast, :, Nb); dims=1)

        if prod(size(C_X2)[1:(N-1)]) == 1
            ΔC_X2_vector = ΔC_X2_scalar
            ΔC_weights = zero(L.C_weights.data)
        else
            ΔC_weights = ΔC_X2_scalar * C_X2_vector'
            ΔC_X2_vector = L.C_weights.data' * ΔC_X2_scalar
        end
        isnothing(L.C_weights.grad) ? (L.C_weights.grad = ΔC_weights) : (L.C_weights.grad += ΔC_weights)
        ΔC_X2 = reshape(ΔC_X2_vector, size(C_X2))
    else
        ΔTm_1 = ΔTm
        ΔC_X2 = zero(C_X2)
    end
    Δw1 = apply_backward(L.scale_activation, ΔSm, w1, Sm)
    Δw2 = apply_backward(L.shift_activation, ΔTm_1, w2, Tm_1)

    # Join scale and shift parts.
    if size(w)[1:N-1] == size(X1)[1:N-1]
        Δw = Δw1 .+ Δw2
    else
        Δw = tensor_cat(Δw1, Δw2)
    end

    return ΔX1, ΔC_X2, Δw
end

scale_logdet_forward(S) = sum(log.(abs.(S)))
scale_logdet_backward(S) = 1f0./ S
