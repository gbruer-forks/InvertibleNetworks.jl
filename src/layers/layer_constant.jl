export LayerConstant

struct LayerConstant <: NeuralNetLayer
    val::Parameter
    logdet::Bool
end

@Flux.functor LayerConstant

function LayerConstant(val::Parameter; logdet=false)
    return LayerConstant(val, logdet)
end

function LayerConstant(val; logdet=false)
    return LayerConstant(Parameter(val); logdet)
end

function LayerConstant(;logdet=false)
    return LayerConstant(nothing; logdet)
end

function forward(X::AbstractArray{T, N}, L::LayerConstant) where {T,N}
    Y = repeat(L.val.data, ones(Int, N-1)..., size(X, N))
    L.logdet == true ? (return Y, zero(T)) : (return Y)
end

function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::LayerConstant) where {T,N}
    Y = repeat(L.val.data, ones(Int, N-1)..., size(X, N))
    L.logdet == true ? (return Y, zero(T)) : (return Y)
end

function backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, L::LayerConstant; set_grad::Bool=true) where {T, N}
    Δval = dropdims(sum(ΔY; dims=N), dims=N)
    if set_grad
        L.val.grad = Δval
        return zero(X)
    end
    return zero(X), [Parameter(Δval)]
end

function backward(ΔY::AbstractArray{T, N}, ΔD::AbstractArray{T, N}, X0::AbstractArray{T, N}, D::AbstractArray{T, N}, L::LayerConstant; set_grad::Bool=true) where {T, N}
    Δval = dropdims(sum(ΔY; dims=N), dims=N)
    if set_grad
        L.val.grad = Δval
        return zero(X)
    end
    return zero(X), [Parameter(Δval)]
end
