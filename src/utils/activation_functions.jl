# Activation functions
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ReLU, ReLUgrad
export LeakyReLU, LeakyReLUinv, LeakyReLUgrad
export Sigmoid, SigmoidInv, SigmoidGrad
export GaLU, GaLUgrad
export ExpClamp, ExpClampInv, ExpClampGrad
export ReLUlayer, LeakyReLUlayer, SigmoidLayer, Sigmoid2Layer, GaLUlayer, ExpClampLayer
export IdentityActivation, SoftplusLayer, TanhLayer, CoshLayer, SinhLayer, DampedCoshLayer, DampedSinhLayer, ScaledTanhLayer
export apply_backward


###############################################################################
# Custom type for activation functions

struct ActivationFunction
    forward::Function
    inverse::Union{Nothing, Function}
    backward::Function
end

"""
Helper function for cases where the caller does not know if the activation function is invertible.
"""
function apply_backward(activation::ActivationFunction, Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    apply_backward(activation.backward, activation.inverse, Δy, x, y)
end

function apply_backward(backward::Function, inverse::Nothing, Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    backward(Δy, x)
end

function apply_backward(backward::Function, inverse::Function, Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    backward(Δy, y)
end

function forward(x::AbstractArray{T, N}, activation::ActivationFunction) where {T, N}
    activation.forward(x)
end

IdentityActivation() = ActivationFunction(identity, identity, IdentityGrad)
IdentityGrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N} = Δy

function ReLUlayer()
    return ActivationFunction(ReLU, nothing, ReLUgrad)
end

function LeakyReLUlayer()
    return ActivationFunction(LeakyReLU, LeakyReLUinv, LeakyReLUgrad)
end

function SigmoidLayer(;low=0f0, high=1f0)
    fwd_a(x) = Sigmoid(x; low=low, high=high)
    inv_a(y) = SigmoidInv(y; low=low, high=high)
    grad_a(Δy, y; x=nothing) = SigmoidGrad(Δy, y; x=x, low=low, high=high)
    return ActivationFunction(fwd_a, inv_a, grad_a)
end

function Sigmoid2Layer()
    fwd_a(x) = 2f0*Sigmoid(x)
    inv_a(y) = SigmoidInv(y/2f0)
    grad_a(Δy, y; x=nothing) = SigmoidGrad(Δy*2f0, y/2f0; x=x)
    return ActivationFunction(fwd_a, inv_a, grad_a)
end

function GaLUlayer()
    return ActivationFunction(GaLU, nothing, GaLUgrad)
end

function ExpClampLayer()
    return ActivationFunction(x -> 2 * ExpClamp(x), y -> ExpClampInv(y/2), (Δy, y) -> ExpClampGrad(Δy*2, y/2))
end


###############################################################################
# Rectified linear unit (ReLU) (not invertible)

"""
    y = ReLU(x)

 Rectified linear unit (not invertible).

 See also: [`ReLUgrad`](@ref)
"""
ReLU(x::AbstractArray{T, N}) where {T, N} =  relu.(x)

"""
    Δx = ReLUgrad(Δy, x)

 Backpropagate data residual through ReLU function.

 *Input*:

 - `Δy`: data residual

 - `x`: original input (since not invertible)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`ReLU`](@ref)
"""
ReLUgrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N} = _relugrad.(Δy, x)

_relugrad(Δy, x) = ifelse(x < 0, zero(x), Δy)

###############################################################################
# Leaky ReLU (invertible)

"""
    y = LeakyReLU(x; slope=0.01f0)

 Leaky rectified linear unit.

 See also: [`LeakyReLUinv`](@ref), [`LeakyReLUgrad`](@ref)
"""
LeakyReLU(x::AbstractArray{T, N}; slope=T(0.01)) where {T, N} = leakyrelu.(x, slope)

"""
    x = LeakyReLUinv(y; slope=0.01f0)

 Inverse of leaky ReLU.

 See also: [`LeakyReLU`](@ref), [`LeakyReLUgrad`](@ref)
"""
LeakyReLUinv(y::AbstractArray{T, N}; slope=T(0.01)) where {T, N} = _lreluinv.(y, slope)

_lreluinv(y::T, slope=T(0.01)) where T = ifelse(y < 0, y/slope, y)

"""
    Δx = LeakyReLUgrad(Δy, x; slope=0.01f0)

 Backpropagate data residual through leaky ReLU function.

 *Input*:

 - `Δy`: data residual

 - `x`: original input (since not invertible)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`LeakyReLU`](@ref), [`LeakyReLUinv`](@ref)
"""

"""
    Δx = ReLUgrad(Δy, y; slope=0.01f0)

 Backpropagate data residual through leaky ReLU function.

 *Input*:

 - `Δy`: residual

 - `y`: original output

 - `slope`: slope of non-active part of ReLU

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`LeakyReLU`](@ref), [`LeakyReLUinv`](@ref)
"""
LeakyReLUgrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; slope=T(0.01)) where {T, N} = _lrelugrad.(Δy, y, slope)

_lrelugrad(Δy::T, y::T, slope=T(0.01)) where T = ifelse(_lreluinv(y, slope) < 0, Δy*slope, Δy)

###############################################################################
# Sigmoid (invertible if ouput nonzero)

"""
    y = Sigmoid(x; low=0, high=1)

 Sigmoid activation function. Shifted and scaled such that output is [low,high].

 See also: [`SigmoidInv`](@ref), [`SigmoidGrad`](@ref)
"""
Sigmoid(x::AbstractArray{T, N}; low=0f0, high=1f0) where {T, N} = _sigmoid.(x, low, high)

_sigmoid(x::T, low=T(0), high=T(1)) where T = high/(1+exp(-x)) + low/(1+exp(x))

"""
    x = SigmoidInv(y; low=0, high=1)

 Inverse of Sigmoid.

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""


"""
    x = SigmoidInv(y; low=0, high=1f0)

 Inverse of Sigmoid function. Shifted and scaled such that output is [low,high]

 See also: [`Sigmoid`](@ref), [`SigmoidGrad`](@ref)
"""
_sigmoidinv(y::T, low=T(0), high=T(1)) where T = log(y - low) - log(high - y)

function SigmoidInv(y::AbstractArray{T, N}; low=0f0, high=1f0) where {T, N}
    if sum(isapprox.(y, 0f-6)) == 0
        return _sigmoidinv.(y, low, high)
    else
        throw(DomainError("Input contains zeros."))
    end
end

"""
    Δx = SigmoidGrad(Δy, y; x=nothing, low=nothing, high=nothing)

 Backpropagate data residual through Sigmoid function. Can be shifted and scaled such that output is (low,high]

 *Input*:

 - `Δy`: residual

 - `y`: original output

 - `x`: original input, if y not available (in this case, set y=nothing)

 - `low`: if provided then scale and shift such that output is (low,high]

 - `high`: if provided then scale and shift such that output is (low,high]

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`Sigmoid`](@ref), [`SigmoidInv`](@ref)
"""
SigmoidGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; x=nothing, low=0f0, high=1f0) where {T, N} = _sigmoidgrad.(x, Δy, y, low, high)
SigmoidGrad(Δy::AbstractArray{T, N}, ::Nothing; x=nothing, low=0f0, high=1f0) where {T, N} = _sigmoidgrad.(x, Δy, nothing, low, high)

_sigmoidgrad(::Nothing, Δy::T, y::T, low=T(0), high=T(1)) where T = _sigmoidgrad(_sigmoidinv(y, low, high), Δy, y, low, high)
_sigmoidgrad(x::T, Δy::T, y, low=T(0), high=T(1)) where T = (high - low) * Δy * exp(-x) / (1 + exp(-x))^2

###############################################################################
# Gated linear unit (GaLU) (not invertible)
# Adapted from Dauphin et al. (2017)

"""
    y = GaLU(x)

 Gated linear activation unit (not invertible).

 See also: [`GaLUgrad`](@ref)
"""
@inline function GaLU(x::AbstractArray{T, N}) where {T, N}
    x1, x2 = tensor_split(x)
    return x1 .* Sigmoid(x2)
end

"""
    Δx = GaLUgrad(Δy, x)

 Backpropagate data residual through GaLU activation.

 *Input*:

 - `Δy`: residual

 - `x`: original input (since not invertible)

 *Output*:

 - `Δx`: backpropagated residual

 See also: [`GaLU`](@ref)
"""
function GaLUgrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    k = Int(size(x, N-1) / 2)
    x1, x2 = tensor_split(x)
    Δx = 0 .*x
    return tensor_cat(Sigmoid(x2) .* Δy, SigmoidGrad(Δy, nothing; x=x2) .* x1)
end

function GaLUjacobian(Δx::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    k = Int(size(x, 3) / 2)
    x1, x2 = tensor_split(x)
    Δx1, Δx2 = tensor_split(Δx)
    s = Sigmoid(x2)
    Δs = SigmoidGrad(Δx2, nothing; x=x2)
    y = x1 .* s
    Δy = Δx1 .* s + x1 .* Δs
    return Δy, y
end

###############################################################################
# Soft-clamped exponential function

"""
    y = ExpClamp(x)
 Soft-clamped exponential function.
 See also: [`ExpClampGrad`](@ref)
"""
ExpClamp(x::AbstractArray{T, N}; clamp=T(2)) where {T, N} = exp.(clamp * T(0.636) * atan.(x))

"""
    x = ExpClampInv(y)
 Inverse of ExpClamp function.
 See also: [`ExpClamp`](@ref), [`ExpClampGrad`](@ref)
"""
function ExpClampInv(y::AbstractArray{T, N}; clamp=T(2)) where {T, N}
    if any(y .≈ 0)
        throw(DomainError("Input contains zeros."))
    else
        return tan.(log.(y) / clamp / T(0.636))
    end
end

"""
    Δx = ExpClampGrad(Δy, x; y=nothing)
 Backpropagate data residual through soft-clamped exponential function.
 *Input*:
 - `Δy`: residual
 - `x`: original input
 *Output*:
 - `Δx`: backpropagated residual
 See also: [`ExpClamp`](@ref)
"""

function ExpClampGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; x=nothing, clamp=T(2)) where {T, N}
    if isnothing(x)
        x = ExpClampInv(y)  # recompute forward state
    end
    return clamp * T(0.636) * Δy .* y ./ (1 .+ x.^2)
end

ExpClampGrad(Δy::AbstractArray{T, N}, ::Nothing; x=nothing, clamp=T(2)) where {T, N} = clamp * T(0.636) * Δy .* y ./ (1 .+ x.^2)



SoftplusLayer() = ActivationFunction(Softplus, SoftplusInv, SoftplusGrad)

function Softplus(x::AbstractArray{T, N}) where {T, N}
    return ifelse.(x .> 20, x, log.(1 .+ exp.(x)))
end

function SoftplusInv(y::AbstractArray{T, N}) where {T, N}
    if any(y .≈ 0)
        throw(DomainError("Input contains zeros."))
    else
        return ifelse.(y .> 20, y, log.(exp.(y) .- 1))
    end
end

function SoftplusGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    return (exp.(y) .- 1) ./ exp.(y) .* Δy
end

struct ScaledTanhLayer{S}
    scale::S
end

function forward(X::AbstractArray{T, N}, L::ScaledTanhLayer{S}) where {T, N, S}
    return Tanh(X) .* T.(L.scale)
end

function inverse(Y::AbstractArray{T, N}, L::ScaledTanhLayer{S}) where {T, N, S}
    return TanhInv(Y ./ T.(L.scale))
end

function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::ScaledTanhLayer{S}) where {T, N, S}
    dY_dtanhX = T.(L.scale)
    tanhX = Y ./ dY_dtanhX
    ΔtanhX = dY_dtanhX .* ΔY
    ΔY = TanhGrad(ΔtanhX, tanhX)
    return ΔY
end

function apply_backward(L::ScaledTanhLayer{S}, Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N, S}
    backward(Δy, y, L)
end


TanhLayer() = ActivationFunction(Tanh, TanhInv, TanhGrad)

function Tanh(x::AbstractArray{T, N}) where {T, N}
    return tanh.(x)
end

function TanhInv(y::AbstractArray{T, N}) where {T, N}
    if any(abs.(y) .> 1 - 1f-6)
        throw(DomainError("Input outside tanh range."))
    else
        return atanh.(y)
    end
end

function TanhGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    return (1 .- y .^ 2) .* Δy
end


CoshLayer() = ActivationFunction(Cosh, nothing, CoshGrad)

function Cosh(x::AbstractArray{T, N}) where {T, N}
    return cosh.(x)
end

function CoshGrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}) where {T, N}
    return sinh.(x) .* Δy
end


DampedCoshLayer() = ActivationFunction(DampedCosh, nothing, DampedCoshGrad)

function DampedCosh(x::AbstractArray{T, N}; a::T=T(10)) where {T, N}
    z0 = x ./ a
    z1 = Tanh(z0)
    z2 = z1 .* a
    y = Cosh(z2)
    return y
end

function DampedCoshGrad(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}; a::T=T(10)) where {T, N}
    z0 = x ./ a
    z1 = Tanh(z0)
    z2 = z1 .* a
    y = Cosh(z2)
    Δz2 = CoshGrad(Δy, z2)
    Δz1 = Δz2 .* a
    Δz0 = TanhGrad(Δz1, z1)
    Δx = Δz0 ./ a
    return Δx
end

SinhLayer() = ActivationFunction(Sinh, SinhInv, SinhGrad)

function Sinh(x::AbstractArray{T, N}) where {T, N}
    return sinh.(x)
end

function SinhInv(y::AbstractArray{T, N}) where {T, N}
    return asinh.(y)
end

function SinhGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    return cosh.(asinh.(y)) .* Δy
end

DampedSinhLayer() = ActivationFunction(DampedSinh, DampedSinhInv, DampedSinhGrad)

function DampedSinh(x::AbstractArray{T, N}; a::T = T(10)) where {T, N}
    z0 = x ./ a
    z1 = Tanh(z0)
    z2 = z1 .* a
    y = Sinh(z2)
    return y
end

function DampedSinhInv(y::AbstractArray{T, N}; a::T = T(10)) where {T, N}
    z2 = SinhInv(y)
    z1 = z2 ./ a
    z0 = TanhInv(z1)
    x = z0 .* a
    return x
end

function DampedSinhGrad(Δy::AbstractArray{T, N}, y::AbstractArray{T, N}; a::T = T(10)) where {T, N}
    z2 = SinhInv(y)
    z1 = z2 ./ a
    z0 = TanhInv(z1)
    x = z0 .* a
    Δz2 = SinhGrad(Δy, y)
    Δz1 = Δz2 .* a
    Δz0 = TanhGrad(Δz1, z1)
    Δx = Δz0 ./ a
    return Δx
end