
export RQSpline1

abstract type RQSpline1{LD} end

struct RQSpline1_params{LD,C} <: RQSpline1{LD}
    x0::Parameter
    y0::Parameter
    d::Parameter
end

struct RQSpline1_func{LD,C} <: RQSpline1{LD}
end

function RQSpline1(; constrained_params=true, with_params=true, logdet=true)
    if with_params
        return RQSpline1_params{logdet,constrained_params}(Parameter(nothing), Parameter(nothing), Parameter(nothing))
    else
        return RQSpline1_func{logdet,constrained_params}()
    end
end

function forward(X::AbstractArray{T, N}, L::RQSpline1_params{LD,C}) where {T,N,LD,C}
    if isnothing(L.x0.data)
        if C
            L.x0.data = fill(T(0), size(X)[1:N-1])
            L.y0.data = fill(T(0), size(X)[1:N-1])
            L.d.data = fill(T(0), size(X)[1:N-1])
        else
            L.x0.data = fill(T(0.5), size(X)[1:N-1])
            L.y0.data = fill(T(0.5), size(X)[1:N-1])
            L.d.data = fill(T(1), size(X)[1:N-1])
        end
    end
    return forward(X, L.x0.data, L.y0.data, L.d.data, RQSpline1_func{LD,C}())
end

function inverse(Y::AbstractArray{T, N}, L::RQSpline1_params{LD,C}) where {T,N,LD,C}
    return inverse(Y, L.x0.data, L.y0.data, L.d.data, RQSpline1_func{LD,C}())
end

function backward(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N}, L::RQSpline1_params{LD,C}) where {T,N,LD,C}
    Δx, Δx0, Δy0, Δd = backward(X, ΔY, L.x0.data, L.y0.data, L.d.data, RQSpline1_func{LD,C}())

    Δx0 = dropdims(sum(Δx0; dims=N); dims=N)
    Δy0 = dropdims(sum(Δy0; dims=N); dims=N)
    Δd = dropdims(sum(Δd; dims=N); dims=N)
    L.x0.grad = Δx0
    L.y0.grad = Δy0
    L.d.grad = Δd
    return Δx
end

function forward(X::AbstractArray{T, N}, x0, y0, d, L::RQSpline1_func{LD,C}) where {T,N,LD,C}
    if C
        x0 = Sigmoid(x0; low=T(0), high=T(1))
        y0 = Sigmoid(y0; low=T(0), high=T(1))
        d = ExpClamp(d; clamp=T(3))
        # d = exp.(T(3) * T(0.636) * atan.(d))
    end
    if LD
        # println()
        # println()
        # # @show size(X) size(x0) size(y0) size(d)
        # println()
        # println()
        # @show size(X)
        # @show size(x0)
        # @show size(y0)
        # @show size(d)
        dy_dxs_ys = spline_derivative.(X, x0, y0, d)
        Y = reshape(getindex.(dy_dxs_ys, 2), size(X))
        logdet = sum(log.(first.(dy_dxs_ys))) ./ size(X, N)
        return Y, logdet
    end
    Y = spline.(X, x0, y0, d)
    return Y
end

function inverse(Y::AbstractArray{T, N}, x0, y0, d, L::RQSpline1_func{LD,C}) where {T,N,LD,C}
    if C
        x0 = Sigmoid(x0; low=T(0), high=T(1))
        y0 = Sigmoid(y0; low=T(0), high=T(1))
        d = ExpClamp(d; clamp=T(3))
    end
    X = spline_inverse.(Y, x0, y0, d)
    return X
end

function backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, x0, y0, d, L::RQSpline1_func{LD,C}) where {T,N,LD,C}
    if LD
        Δlogdet = T(-1)
    else
        Δlogdet = T(0)
    end
    return backward(ΔY, Δlogdet, X, x0, y0, d, L)
end

function backward(ΔY::AbstractArray{T, N}, Δlogdet::T, X::AbstractArray{T, N}, x0, y0, d, L::RQSpline1_func{LD,C}) where {T,N,LD,C}
    if C
        x0_orig = x0
        y0_orig = y0
        d_orig = d
        x0 = Sigmoid(x0; low=T(0), high=T(1))
        y0 = Sigmoid(y0; low=T(0), high=T(1))
        d = ExpClamp(d; clamp=T(3))
    end
    # X_dy_dx = spline_inverse.(Y, x0, y0, d; derivative=true)
    # X = first.(X_dy_dx)
    # dy_dx = getindex.(X_dy_dx, 2)

    dy_dx_Y = spline_derivative.(X, x0, y0, d)
    dy_dx = first.(dy_dx_Y)
    Y = getindex.(dy_dx_Y, 2)

    # @show Δlogdet
    # error("done")
    Δdy_dx = (Δlogdet / size(Y, N)) ./ dy_dx
    Δx_Δx0_Δy0_Δd = Broadcast.broadcasted(spline_adjoint, X, x0, y0, d, ΔY, Δdy_dx)
    Δx = getindex.(Δx_Δx0_Δy0_Δd, 1)
    Δx0 = getindex.(Δx_Δx0_Δy0_Δd, 2)
    Δy0 = getindex.(Δx_Δx0_Δy0_Δd, 3)
    Δd = getindex.(Δx_Δx0_Δy0_Δd, 4)

    if C
        if size(Δx0) != size(x0)
            # Do dummy operation to make the size the same as a broadcasted operation would.
            x0 = x0 .+ zeros(size(Δx0))
            x0_orig = x0_orig .+ zeros(size(Δx0))
        end
        if size(Δy0) != size(y0)
            # Do dummy operation to make the size the same as a broadcasted operation would.
            y0 = y0 .+ zeros(size(Δy0))
            y0_orig = y0_orig .+ zeros(size(Δy0))
        end
        if size(Δd) != size(d)
            # Do dummy operation to make the size the same as a broadcasted operation would.
            d = d .+ zeros(size(Δd))
            d_orig = d_orig .+ zeros(size(Δd))
        end
        Δx0 = SigmoidGrad(Δx0, x0; x=x0_orig, low=T(0), high=T(1))
        Δy0 = SigmoidGrad(Δy0, y0; x=y0_orig, low=T(0), high=T(1))
        Δd = ExpClampGrad(Δd, d; x=d_orig, clamp=T(3))
    end
    return Δx, Δx0, Δy0, Δd
end


"""RQ-spline defined by one point within (0,0) to (1,1) with slope d at that point, and slope y = x elsewhere."""
function spline(x, x0, y0, d)
    if x <= 0 || x >= 1
        return x
    end
    if x < x0
        return spline_bin(x, x0, y0, d)
    end
    x = 1 - x
    x0 = 1 - x0
    y0 = 1 - y0
    return 1 - spline_bin(x, x0, y0, d)
end

"""Evaluates the RQ-spline assuming x ∈ [0, x0] for spline with y=x for x < 0 and slope d at (x0, y0)."""
function spline_bin(x, x0, y0, d)
    if y0 == 0
        return y0
    end
    zx = x / x0
    d1 = d * x0 / y0
    d0 = x0 / y0
    zy, _, _ = spline_bin_normalized(zx, d0, d1)
    return zy * y0
end

"""Evaluates the RQ-spline assuming x ∈ [0, 1] for spline with y=d0*x for x < 0 and slope d1 at (1, 1)."""
function spline_bin_normalized(zx, d0, d1)
    zx_1mzx = zx * (1 - zx)
    den = 1 + (d0 + d1 - 2) * zx_1mzx
    zy = (zx ^ 2 + d0 * zx_1mzx) / den
    return zy, zx_1mzx, den
end

"""Evaluates the derivative of the RQ-spline assuming x ∈ [0, 1] for spline with y=d0*x for x < 0 and slope d1 at (1, 1)."""
function spline_bin_normalized_derivative(zx, d0, d1)
    zy, zx_1mzx, den = spline_bin_normalized(zx, d0, d1)
    dzy_dzx = (2 * (1 - d0) * zx + d0 - zy * (d0 + d1 - 2) * (1 - 2*zx)) / den
    return dzy_dzx, zy, zx_1mzx, den
end

"""Evaluates the derivative of the RQ-spline assuming x ∈ [0, x0] for spline with y=x for x < 0 and slope d at (x0, y0)."""
function spline_bin_derivative(x, x0, y0, d)
    if y0 == 0
        return zero(x), y0
    end
    zx = x / x0
    d0 = x0 / y0
    d1 = d * d0
    dzy_dzx, zy, _, _ = spline_bin_normalized_derivative(zx, d0, d1)
    y = zy * y0
    dx_dzx = x0
    dy_dzy = y0
    return dy_dzy * dzy_dzx / dx_dzx, y
end

"""Derivative of RQ-spline defined by one point (x0, y0) within (0,0) to (1,1) with slope d at that point, and slope y = x elsewhere."""
function spline_derivative(x, x0, y0, d)
    if x <= 0 || x >= 1
        return one(x), x
    end
    if x < x0
        return spline_bin_derivative(x, x0, y0, d)
    end
    x = 1 - x
    x0 = 1 - x0
    y0 = 1 - y0
    dnx_dny, y = spline_bin_derivative(x, x0, y0, d)
    return dnx_dny, 1 - y
end

"""Applies the adjoint Jacobian applied to the given values for the RQ-spline assuming x ∈ [0, 1] for spline with y=d0*x for x < 0 and slope d1 at (1, 1)."""
function spline_bin_normalized_adjoint(zx, d0, d1, Δzy, Δdzy_dzx=0)
    dzy_dzx, zy, zx_1mzx, den = spline_bin_normalized_derivative(zx, d0, d1)

    # d_dzydzx_d_zx = (2 * (1 - d0) + 2 * zy * (d0 + d1 - 2) - dzy_dzx * (d0 + d1 - 2) * (1 - 2 * zx)) / den
    # d_dzydzx_d_zx = (2(1 - d0) + (-2(2 - d0 - d1)*(zx^2 + d0*(1 - zx)*zx) + (2 - d0 - d1)*(2zx - d0*zx + d0*(1 - zx))*(1 - 2zx)) / (1 + (-2 + d0 + d1)*(1 - zx)*zx) - ((-2 + d0 + d1)*(1 - zx) - (-2 + d0 + d1)*zx)*((-(-2 + d0 + d1)*(zx^2 + d0*(1 - zx)*zx)*(1 - 2zx)) / ((1 + (-2 + d0 + d1)*(1 - zx)*zx)^2))) / (1 + (-2 + d0 + d1)*(1 - zx)*zx) - ((-2 + d0 + d1)*(1 - zx) - (-2 + d0 + d1)*zx)*((d0 + (-(-2 + d0 + d1)*(zx^2 + d0*(1 - zx)*zx)*(1 - 2zx)) / (1 + (-2 + d0 + d1)*(1 - zx)*zx) + 2(1 - d0)*zx) / ((1 + (-2 + d0 + d1)*(1 - zx)*zx)^2))
    d_dzydzx_d_zx = 2 * (1 - d0 + (d0 + d1 - 2)*(zy - dzy_dzx * (1 - 2 * zx))) / den
    # d_dzydzx_d_zx = (2 * (1 - d0) + (2 * zy - dzy_dzx * (1 - 2 * zx)) * (d0 + d1 - 2)) / den

    # d_dzydzx_d_d0 = ((1 - zy) * (1 - 2 * zx) - dzy_dzx * zx_1mzx) / den
    # d_dzydzx_d_d1 = (-zy * (1 - 2 * zx) - dzy_dzx * zx_1mzx) / den
    m2zx_p1 = 1 - 2 * zx
    dzy_dd0 = (1 - zy) * zx_1mzx / den
    dzy_dd1 = -zy * zx_1mzx / den
    d_dzydzx_d_d0 = (m2zx_p1 * (1 - zy) - dzy_dzx * zx_1mzx - dzy_dd0 * (d0 + d1 - 2) * m2zx_p1) / den
    d_dzydzx_d_d1 = - (zy * m2zx_p1 + dzy_dzx * zx_1mzx + dzy_dd1 * (d0 + d1 - 2) * m2zx_p1) / den

    Δzx = dzy_dzx * Δzy + d_dzydzx_d_zx * Δdzy_dzx
    Δd0 = dzy_dd0 * Δzy + d_dzydzx_d_d0 * Δdzy_dzx
    Δd1 = dzy_dd1 * Δzy + d_dzydzx_d_d1 * Δdzy_dzx
    return Δzx, Δd0, Δd1, zy, dzy_dzx
end

"""Applies the adjoint Jacobian to the given delta for the RQ-spline assuming x ∈ [0, x0] for spline with y=x for x < 0 and slope d at (x0, y0)."""
function spline_bin_adjoint(x, x0, y0, d, Δy, Δdy_dx=0)
    # if y0 == 0
    #     # return zero(x)
    # end

    dx_dzx = x0
    dy_dzy = y0

    zx = x / x0
    d0 = x0 / y0
    d1 = d * d0

    Δzy = dy_dzy * Δy

    d_dydx_d_dzydzx = dy_dzy / dx_dzx
    Δdzy_dzx = Δdy_dx * d_dydx_d_dzydzx

    Δzx, Δd0, Δd1, zy, dzy_dzx = spline_bin_normalized_adjoint(zx, d0, d1, Δzy, Δdzy_dzx)

    dy_dy0 = zy

    dd0_dx0 = 1 / y0
    dd0_dy0 = -x0 / y0 ^ 2

    dd1_dd = d0
    dd1_dx0 = d * dd0_dx0
    dd1_dy0 = d * dd0_dy0

    d_dydx_d_dx0 = - dy_dzy * dzy_dzx / x0 ^ 2
    d_dydx_d_dy0 = dzy_dzx / dx_dzx

    dzx_dx0 = - x / x0 ^ 2

    Δd = dd1_dd * Δd1
    Δx0 = dd1_dx0 * Δd1 + dd0_dx0 * Δd0 + dzx_dx0 * Δzx + d_dydx_d_dx0 * Δdy_dx
    Δy0 = dy_dy0 * Δy + dd1_dy0 * Δd1 + dd0_dy0 * Δd0 + d_dydx_d_dy0 * Δdy_dx

    Δx = Δzx / dx_dzx
    return Δx, Δx0, Δy0, Δd
end

"""Applies the adjoint Jacobian for RQ-spline defined by one point (x0, y0) within (0,0) to (1,1) with slope d at that point, and slope y = x elsewhere."""
function spline_adjoint(x, x0, y0, d, Δy, Δdy_dx=0)
    if x <= 0 || x >= 1
        return Δy, zero(x0), zero(y0), zero(d)
    end
    if x < x0
        return spline_bin_adjoint(x, x0, y0, d, Δy, Δdy_dx)
    end
    x = 1 - x
    x0 = 1 - x0
    y0 = 1 - y0
    Δy = -Δy
    Δnx, Δnx0, Δny0, Δd = spline_bin_adjoint(x, x0, y0, d, Δy, Δdy_dx)
    return -Δnx, -Δnx0, -Δny0, Δd
end

"""Evaluates the inverse of the RQ-spline assuming x ∈ [0, 1] for spline with y=d0*x for x < 0 and slope d1 at (1, 1)."""
function spline_bin_normalized_inverse(zy, d0, d1; derivative=false)
    a = 1 - d0 - zy * (2 - d0 - d1)
    b = d0 + zy * (2 - d0 - d1)
    c = -zy
    zx = 2 * c / (-b - sqrt(b ^ 2 - 4 * a * c))
    if derivative
        den = 1 + (d0 + d1 - 2) * zx * (1 - zx)
        dzy_dzx = (2 * (1 - d0) * zx + d0 - zy * (d0 + d1 - 2) * (1 - 2*zx)) / den
        return zx, dzy_dzx
    end
    return zx
end

"""Evaluates the inverse of the RQ-spline assuming x ∈ [0, x0] for spline with y=x for x < 0 and slope d at (x0, y0)."""
function spline_bin_inverse(y, x0, y0, d; derivative=false)
    if y0 == 0
        if derivative
            return y0, zero(y)
        else
            return y0
        end
    end
    zy = y / y0
    d1 = d * x0 / y0
    d0 = x0 / y0
    if derivative
        dx_dzx = x0
        dy_dzy = y0
        zx, dzy_dzx = spline_bin_normalized_inverse(zy, d0, d1; derivative)
        return zx * x0, dy_dzy * dzy_dzx / dx_dzx
    else
        zx = spline_bin_normalized_inverse(zy, d0, d1; derivative)
        return zx * x0
    end
end

"""Inverse of RQ-spline defined by one point within (0,0) to (1,1) with slope d at that point, and slope y = x elsewhere."""
function spline_inverse(y, x0, y0, d; derivative=false)
    if y < 0 || y > 1
        if derivative
            return y, one(y)
        else
            return y
        end
    end
    if y < y0
        return spline_bin_inverse(y, x0, y0, d; derivative)
    end
    y = 1 - y
    x0 = 1 - x0
    y0 = 1 - y0
    if derivative
        x, dnx_dny = spline_bin_inverse(y, x0, y0, d; derivative)
        x = 1 - x
        return x, dnx_dny
    else
        return 1 - spline_bin_inverse(y, x0, y0, d; derivative)
    end
end
