export LayerAffineActNorm

struct LayerAffineActNorm <: InvertibleNetwork
    AN::ActNorm
    A::Affine
    activation::ActivationFunction
end

@Flux.functor LayerAffineActNorm

# Constructor
function LayerAffineActNorm(in_shape, cond_shape, L, K;
)
    return LayerAffineActNorm(state_networks, state_final_network, cond_network, CL, Z_dims, L, K, squeezer, split_scales)
end

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::LayerAffineActNorm) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, G.L-1))
    orig_shape = size(X)

    if !isnothing(G.cond_network)
        C = forward(C, G.cond_network)
    end

    logdet = 0
    for i=1:G.L
        (G.split_scales) && (X = forward(X, G.squeezer))
        (G.split_scales) && (C = forward(C, G.squeezer))
        for j=1:G.K            
            if !isnothing(G.state_networks[i, j])
                X, logdet1 = forward(X, G.state_networks[i, j])
                logdet += logdet1
            end
            X, logdet2 = forward(X, C, G.CL[i, j])
            logdet += logdet2
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = reshape(cat_states(Z_save, X),orig_shape))
    if !isnothing(G.state_final_network)
        X, logdet1 = forward(X, G.state_final_network)
        logdet += logdet1
    end
    return X, C, logdet
end

# Inverse pass 
function inverse(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::LayerAffineActNorm) where {T, N}
    if !isnothing(G.state_final_network)
        X = inverse(X, G.state_final_network)
    end
    # if !isnothing(G.cond_network)
    #     C = forward(C, G.cond_network)
    # end
    G.split_scales && ((Z_save, X) = split_states(X[:], G.Z_dims))
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            X = inverse(X, C, G.CL[i, j])
            if !isnothing(G.state_networks[i, j])
                X = inverse(X, G.state_networks[i, j])
            end
        end

        (G.split_scales) && (X = inverse(X, G.squeezer))
        (G.split_scales) && (C = inverse(C, G.squeezer))
    end
    return X
end

# Backward pass and compute gradients
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::LayerAffineActNorm;) where {T, N}
    # Split data and gradients
    if G.split_scales
        ΔZ_save, ΔX = split_states(ΔX[:], G.Z_dims)
        Z_save, X = split_states(X[:], G.Z_dims)
    end

    if !isnothing(G.state_final_network)
        ΔX, X = backward(ΔX, X, G.state_final_network)
    end

    ΔC = T(0) .* C
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X  = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=G.K:-1:1
            ΔX, X, ΔC_ = backward(ΔX, X, C, G.CL[i, j])
            if !isnothing(G.state_networks[i, j])
                ΔX, X = backward(ΔX, X, G.state_networks[i, j])
            end
            ΔC += ΔC_      
        end

        if G.split_scales 
            C = G.squeezer.inverse(C)
            ΔC = G.squeezer.inverse(ΔC) 
            X = G.squeezer.inverse(X)
            ΔX = G.squeezer.inverse(ΔX)
        end
    end

    if !isnothing(G.cond_network)
        ΔC, C = backward(ΔC, C, G.cond_network)
    end
    return ΔX, X, ΔC
end
