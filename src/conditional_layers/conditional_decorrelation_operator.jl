export ConditionalDecorrelationOperator, get_params_shape

struct ConditionalDecorrelationOperator <: NeuralNetLayer
    correlation_activation
    y_activation
end

function ConditionalDecorrelationOperator(; correlation_activation = ScaledTanhLayer(0.999), y_activation=IdentityActivation())
    return ConditionalDecorrelationOperator(correlation_activation, y_activation)
end

function get_params_shape(input_shape, L::ConditionalDecorrelationOperator)
    num_params = input_shape[end] * 2
    params_shape = tuple(input_shape[1:end-1]..., num_params)
    return params_shape
end

function forward(X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{Tw, Nw}, L::ConditionalDecorrelationOperator) where {T,Tw,N,NcNx2,Nw}
    # Split subnetwork output to get correlation and V'y parts.
    w_S, w_y = tensor_split(w)

    # Apply correlation decoupling.
    S = forward(w_S, L.correlation_activation)
    b = forward(w_y, L.y_activation)
    scale = 1 ./ sqrt.(1 .- S .^ 2)
    Y1 = scale .* (X1 .- S .* b)
    logdet = scale_logdet_forward(scale) / size(X1, N)
    return Y1, logdet
end


function inverse(Y1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::ConditionalDecorrelationOperator) where {T,N,NcNx2,Nw}
    # Split subnetwork output to get scale and shift parts.
    w_S, w_y = tensor_split(w)

    # Invert correlation decoupling.
    S = forward(w_S, L.correlation_activation)
    b = forward(w_y, L.y_activation)
    scale = 1 ./ sqrt.(1 .- S .^ 2)
    X1 = Y1 ./ scale + S .* b
    return X1, (; w_S, w_y, S, b, scale)
end

function backward(ΔY1::AbstractArray{T, N}, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::ConditionalDecorrelationOperator, saved) where {T,N,NcNx2,Nw}
    Δlogdet = T(-1) / size(ΔY1, N)
    return backward(ΔY1, Δlogdet, X1, C_X2, w, L, saved)
end

function backward(ΔY1::AbstractArray{T, N}, Δlogdet::T, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::ConditionalDecorrelationOperator, saved) where {T,N,NcNx2,Nw}
    (; w_S, w_y, S, b, scale) = saved

    dY1_dscale = X1 .- S .* b
    dY1_dX1 = scale
    dY1_dS = -scale .* b
    dY1_db = -scale .* S

    dscale_dS = S ./ (1 .- S .^ 2) .^ (3/2)

    dlogdet_dscale = scale_logdet_backward(scale)

    Δscale = dY1_dscale .* ΔY1 .+ dlogdet_dscale .* Δlogdet
    ΔX1 = dY1_dX1 .* ΔY1 
    ΔS = dY1_dS .* ΔY1 .+ dscale_dS .* Δscale
    Δb = dY1_db .* ΔY1

    Δw_S = apply_backward(L.correlation_activation, ΔS, w_S, S)
    Δw_y = apply_backward(L.y_activation, Δb, w_y, b)

    Δw = tensor_cat(Δw_S, Δw_y)

    ΔC_X2 = zero(C_X2)

    return ΔX1, ΔC_X2, Δw
end
