export RQSpline1Operator

struct RQSpline1Operator{LD,C} <: NeuralNetLayer
    spline::RQSpline1_func{LD,C}
end

function RQSpline1Operator(;logdet=true, constrained_params=true)
    return RQSpline1Operator(RQSpline1_func{logdet,constrained_params}())
end

function extract_rqspline1_params(w::AbstractArray{T, N}) where {T, N}
    # Split w along channel dimension into three parts.
    nc = size(w, N-1)
    if mod(nc, 3) != 0
        error("Expected nc $nc to be divisible by 3")
    end
    np = nc ÷ 3
    x0 = selectdim(w, N-1, 1:np)
    y0 = selectdim(w, N-1, (np+1):(2*np))
    d = selectdim(w, N-1, (2*np+1):size(w,N-1))
    # println()
    # @show size(w) size(x0) size(y0) size(d)
    # println()
    return x0, y0, d
end


function unextract_rqspline1_params(x0, y0, d)
    w = cat(x0, y0, d; dims=size(x0)[end-1])
    return w
end

function forward(X1::AbstractArray{T, N}, _::AbstractArray{T, NcNx2}, w::AbstractArray{Tw, Nw}, L::RQSpline1Operator) where {T,Tw,N,NcNx2,Nw}
    x0, y0, d = extract_rqspline1_params(w)
    return forward(X1, x0, y0, d, L.spline)
end


function inverse(Y1::AbstractArray{T, N}, _::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::RQSpline1Operator) where {T,N,NcNx2,Nw}
    x0, y0, d = extract_rqspline1_params(w)
    return inverse(Y1, x0, y0, d, L.spline), nothing
end

function backward(ΔY1::AbstractArray{T, N}, Δlogdet::T, X1::AbstractArray{T, N}, C_X2::AbstractArray{T, NcNx2}, w::AbstractArray{T, Nw}, L::RQSpline1Operator, saved) where {T,N,NcNx2,Nw}
    x0, y0, d = extract_rqspline1_params(w)
    Δx, Δx0, Δy0, Δd, X = backward(ΔY1, X1, x0, y0, d, L.spline)
    Δw = unextract_rqspline1_params(Δx0, Δy0, Δd)
    return Δx, zero(C_X2), Δw
end

