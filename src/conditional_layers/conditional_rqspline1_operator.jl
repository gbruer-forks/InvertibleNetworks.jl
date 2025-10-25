export RQSpline1Operator, get_params_shape

struct RQSpline1Operator{LD,C} <: NeuralNetLayer
    spline::RQSpline1_func{LD,C}
    affine::AffineCouplingOperator
end

function RQSpline1Operator(;logdet=true, constrained_params=true, affine=AffineCouplingOperator())
    return RQSpline1Operator(RQSpline1_func{logdet,constrained_params}(), affine)
end

function get_params_shape(input_shape, L::RQSpline1Operator)
    num_params = input_shape[end] * 3
    params_shape_affine = get_params_shape(input_shape, L.affine)
    if params_shape_affine[1:end-1] != input_shape[1:end-1]
        error("I didn't set this up yet.")
    end
    num_params = num_params + params_shape_affine[end]
    params_shape = tuple(input_shape[1:end-1]..., num_params)
    # @show input_shape params_shape_affine params_shape
    # @show params_shape params_shape_affine input_shape
    return params_shape
end

function extract_rqspline1_params(w::AbstractArray{T, N}, nchan) where {T, N}
    # Split w along channel dimension into parts.
    s,e = 1, nchan
    x0 = selectdim(w, N-1, s:e)

    s,e = e+1, e+nchan
    y0 = selectdim(w, N-1, s:e)

    s,e = e+1, e+nchan
    d = selectdim(w, N-1, s:e)

    s,e = e+1, size(w, N-1)
    w_rest = selectdim(w, N-1, s:e)
    return x0, y0, d, w_rest
end


function unextract_rqspline1_params(x0, y0, d, w_rest)
    w = cat(x0, y0, d, w_rest; dims=ndims(x0)-1)
    return w
end

function forward(X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{Tw, Nw}, L::RQSpline1Operator) where {T,Tw,N,NcNx2,Nw}
    nchan = size(X1, N-1)
    # @show size(X1)
    x0, y0, d, w_rest = extract_rqspline1_params(w, nchan)
    X1_b, logdet1 = forward(X1, C_X2, w_rest, L.affine)
    Y1, logdet2 = forward(X1_b, x0, y0, d, L.spline)
    return Y1, logdet1 + logdet2
end

function inverse(Y1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::RQSpline1Operator) where {T,N,NcNx2,Nw}
    nchan = size(Y1, N-1)
    x0, y0, d, w_rest = extract_rqspline1_params(w, nchan)
    X1_b = inverse(Y1, x0, y0, d, L.spline)
    X1, saved = inverse(X1_b, C_X2, w_rest, L.affine)
    return X1, saved
end

function backward(ΔY1::AbstractArray{T, N}, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::RQSpline1Operator, saved) where {T,N,NcNx2,Nw}
    nchan = size(X1, N-1)
    x0, y0, d, w_rest = extract_rqspline1_params(w, nchan)
    X1_b, _ = forward(X1, C_X2, w_rest, L.affine)
    ΔX1_b, Δx0, Δy0, Δd = backward(ΔY1, X1_b, x0, y0, d, L.spline)
    ΔX1, ΔC_X2, Δw_rest = backward(ΔX1_b, X1, C_X2, w_rest, L.affine, saved)
    Δw = unextract_rqspline1_params(Δx0, Δy0, Δd, Δw_rest)
    return ΔX1, ΔC_X2, Δw
end

function backward(ΔY1::AbstractArray{T, N}, Δlogdet::T, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::RQSpline1Operator, saved) where {T,N,NcNx2,Nw}
    nchan = size(X1, N-1)
    x0, y0, d, w_rest = extract_rqspline1_params(w, nchan)
    X1_b, _ = forward(X1, C_X2, w_rest, L.affine)
    ΔX1_b, Δx0, Δy0, Δd = backward(ΔY1, 0, X1_b, x0, y0, d, L.spline)
    ΔX1, ΔC_X2, Δw_rest = backward(ΔX1_b, 0, X1, C_X2, w_rest, L.affine, saved)
    Δw = unextract_rqspline1_params(Δx0, Δy0, Δd, Δw_rest)
    # @show ΔY1
    # @show ΔX1_b
    # @show ΔX1
    # @show ΔC_X2
    # @show Δw
    return ΔX1, ΔC_X2, Δw
end

