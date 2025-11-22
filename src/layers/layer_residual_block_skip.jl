
export ResidualBlockSkip

ResidualBlockActTypeSkip = Union{ActivationFunction, RQSpline1_params}

"""
Same as ResidualBlock but has an extra set of weights and skip connection such that the output is
proportional to the input when all weights are set to 0.
"""
struct ResidualBlockSkip{T1<:ResidualBlockActTypeSkip, T2<:ResidualBlockActTypeSkip} <: NeuralNetLayer
    W1::Parameter
    W2::Parameter
    W3::Parameter
    W_13::Parameter
    b1::Parameter
    b2::Parameter
    W_13_offset
    strides
    pad
    activation::T1
    final_activation::T2
end

@Flux.functor ResidualBlockSkip

#######################################################################################################################
#  Constructors

# Constructor
function ResidualBlockSkip(n_in, n_hidden; n_out=nothing, activation=ReLUlayer(), k1=3, k2=3, k13=3, p1=1, p2=1, p13=1, s1=1, s2=1, s13=1, ndims=2, final_activation=activation)
    # default/legacy behaviour
    isnothing(n_out) && (n_out = 2*n_in)

    k1 = Tuple(k1 for i=1:ndims)
    k2 = Tuple(k2 for i=1:ndims)
    k13 = Tuple(k13 for i=1:ndims)

    # Initialize weights
    W1 = Parameter(glorot_uniform(k1..., n_in, n_hidden))
    W2 = Parameter(glorot_uniform(k2..., n_hidden, n_hidden))
    W3 = Parameter(glorot_uniform(k1..., n_out, n_hidden))
    W_13 = Parameter(glorot_uniform(k13..., n_in, n_out))
    b1 = Parameter(zeros(Float32, n_hidden))
    b2 = Parameter(zeros(Float32, n_hidden))

    W_13_offset = zeros(eltype(W_13.data), size(W_13.data))
    W_13_offset[Tuple(ceil(Int, k13[1]/2) for i=1:ndims)..., :, :] = Matrix(I, n_in, n_out)
    return ResidualBlockSkip(W1, W2, W3, W_13, b1, b2, W_13_offset, (s1, s2, s13), (p1, p2, p13), activation, final_activation)
end

#######################################################################################################################
# Functions

# Forward
function forward(X1::AbstractArray{T, N}, RB::ResidualBlockSkip; save=false) where {T, N}
    inds =[i!=(N-1) ? 1 : Colon() for i=1:N]

    # First convolution should downsample the image.
    Y1 = conv(X1, RB.W1.data; stride=RB.strides[1], pad=RB.pad[1]) .+ reshape(RB.b1.data, inds...)
    X2 = forward(Y1, RB.activation)

    Y2 = X2 + conv(X2, RB.W2.data; stride=RB.strides[2], pad=RB.pad[2]) .+ reshape(RB.b2.data, inds...)
    X3 = forward(Y2, RB.activation)

    # Include skip connection.
    Y_13 = conv(X1, RB.W_13.data + T.(RB.W_13_offset); stride=RB.strides[3], pad=RB.pad[3])

    # Last convolution should upsample the image to the size of the input, but with a new number of channels.
    cdims3 = DCDims(X1, RB.W3.data; stride=RB.strides[1], padding=RB.pad[1])
    Y3 = Y_13 + ∇conv_data(X3, RB.W3.data, cdims3)


    # Return if only recomputing state
    X4 = forward(Y3, RB.final_activation)
    save && (return Y1, Y2, Y3, Y_13, X2, X3, X4)

    # Finish forward
    return X4
end

# Backward
function backward(ΔX4::AbstractArray{T, N}, X1::AbstractArray{T, N},
                  RB::ResidualBlockSkip; set_grad::Bool=true) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Recompute forward states from input X
    Y1, Y2, Y3, Y_13, X2, X3, X4 = forward(X1, RB; save=true)

    # Cdims
    cdims2 = DenseConvDims(Y2, RB.W2.data; stride=RB.strides[2], padding=RB.pad[2])
    cdims3 = DCDims(X1, RB.W3.data;  stride=RB.strides[1], padding=RB.pad[1])

    # Backpropagate residual ΔX4 and compute gradients
    ΔY3 = apply_backward(RB.final_activation, ΔX4, Y3, X4)
    ΔY_13 = ΔY3
    ΔX3 = conv(ΔY3, RB.W3.data, cdims3)
    ΔW3 = ∇conv_filter(ΔY3, X3, cdims3)

    ΔY2 = apply_backward(RB.activation, ΔX3, Y2, X3)
    ΔX2 = ∇conv_data(ΔY2, RB.W2.data, cdims2) + ΔY2
    ΔW2 = ∇conv_filter(X2, ΔY2, cdims2)
    Δb2 = sum(ΔY2, dims=dims)[inds...]

    cdims1 = DenseConvDims(X1, RB.W1.data; stride=RB.strides[1], padding=RB.pad[1])

    cdims13 = DenseConvDims(X1, RB.W_13.data; stride=RB.strides[3], padding=RB.pad[3])
    ΔX1 = ∇conv_data(ΔY_13, RB.W_13.data + T.(RB.W_13_offset), cdims13)
    ΔW_13 = ∇conv_filter(X1, ΔY_13, cdims13)

    ΔY1 = apply_backward(RB.activation, ΔX2, Y1, X2)
    ΔX1 = ∇conv_data(ΔY1, RB.W1.data, cdims1) + ΔX1
    ΔW1 = ∇conv_filter(X1, ΔY1, cdims1)
    Δb1 = sum(ΔY1, dims=dims)[inds...]

    # Set gradients
    if set_grad
        RB.W1.grad = ΔW1
        RB.W2.grad = ΔW2
        RB.W3.grad = ΔW3
        RB.W_13.grad = ΔW_13
        RB.b1.grad = Δb1
        RB.b2.grad = Δb2
    else
        Δθ = [Parameter(ΔW1), Parameter(ΔW2), Parameter(ΔW3), Parameter(Δb1), Parameter(Δb2)]
    end

    set_grad ? (return ΔX1) : (return ΔX1, Δθ)
end
