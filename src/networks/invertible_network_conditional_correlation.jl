export NetworkConditionalCorrelation

struct NetworkConditionalCorrelation <: InvertibleNetwork
    state_networks::AbstractArray{Union{ActNorm, Nothing}, 2}
    state_final_network::Union{ActNorm, Nothing}
    cond_network::Union{ActNorm, Nothing}
    CL::AbstractArray{ConditionalLayerCorrelation, 2}
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    squeezer::Union{Squeezer, Nothing}
    split_scales::Bool
end

@Flux.functor NetworkConditionalCorrelation

# Constructor
function NetworkConditionalCorrelation(in_shape, cond_shape, L, K;
    split_scales=false,
    cond_network_generator=nothing,
    state_initial_network_generator=nothing,
    state_middle_network_generator=nothing,
    state_final_network_generator=nothing,
    prenetwork_generator=nothing,
    subnetwork_generator=nothing,
    squeezer=nothing,
    coupling_layer_params=(;),
)
    if isnothing(state_initial_network_generator)
        state_initial_network_generator = in_shape -> nothing
    end

    if isnothing(state_middle_network_generator)
        state_middle_network_generator = in_shape -> nothing
    end

    if isnothing(state_final_network_generator)
        state_final_network_generator = in_shape -> nothing
    end

    if isnothing(cond_network_generator)
        cond_network_generator = cond_shape -> nothing
    end

    if isnothing(prenetwork_generator)
        prenetwork_generator = in_shape -> nothing
    end

    if isnothing(subnetwork_generator)
        subnetwork_generator = (in_shape, out_shape) -> LayerConstant(glorot_uniform(out_shape...))
    end

    state_networks = Array{Union{ActNorm, Nothing}}(undef, L, K)    # activation normalization
    cond_network = cond_network_generator(cond_shape)
    CL = Array{ConditionalLayerCorrelation}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
 
    if split_scales
        Z_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        shape_factor = 2
        channel_factor = shape_factor^(length(in_shape) - 1)
        if isnothing(squeezer)
            squeezer = ShuffleLayer()
        end
    else
        Z_dims = nothing
        channel_factor = 1
        shape_factor = 1
        @assert L == 1
    end

    in_shape = collect(in_shape)
    out_shape = collect(in_shape)
    cond_shape = collect(cond_shape)
    for i=1:L
        # squeeze if split_scales is turned on
        # in_shape[1:end-1] = Int64.(in_shape[1:end-1] / 2)
        in_shape[end] *= channel_factor

        # cond_shape[1:end-1] = Int64.(cond_shape[1:end-1] / 2)
        cond_shape[end] *= channel_factor 
        for j=1:K
            if i == 1 && j == 1
                state_networks[i, j] = state_initial_network_generator(in_shape) # ActNorm(in_shape[end]; logdet=true)
            else
                state_networks[i, j] = state_middle_network_generator(in_shape) # ActNorm(in_shape[end]; logdet=true)
            end

            # 1x1 Convolution and residual block for coupling layers
            in_split, split_num = ConditionalLayerCorrelation_splitdims(in_shape[end])
            out_shape[end]  = split_num

            prenetwork = prenetwork_generator(in_shape)
            sub_shape = tuple(in_shape[1:end-1]..., in_split+cond_shape[end])
            subnetwork = subnetwork_generator(sub_shape, out_shape)
            CL[i, j] = ConditionalLayerCorrelation(prenetwork, subnetwork; coupling_layer_params..., logdet=true)
        end
        (i < L && split_scales) && (in_shape[end] = Int64(in_shape[end]/2)) # split
    end
    state_final_network = state_final_network_generator(in_shape)

    return NetworkConditionalCorrelation(state_networks, state_final_network, cond_network, CL, Z_dims, L, K, squeezer, split_scales)
end

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkConditionalCorrelation) where {T, N}
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
function inverse(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkConditionalCorrelation) where {T, N}
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
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkConditionalCorrelation;) where {T, N}
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
