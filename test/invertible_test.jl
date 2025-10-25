
include("grad_test.jl")

function invertible_layer_test_inverse(L, X, dX)
    X_ = inverse(forward(X, L)[1], L)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-5)

    X_ = forward(inverse(X, L), L)[1]
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-5)
end

function invertible_layer_test_gradient(L, P, dP, X, dX; backward_y=true, name, do_flux=nothing, tol=1e-10)
    L_params = get_params(L)
    loss = function (P, X; with_grad)
        if !isnothing(P)
            set_params!(L_params, P)
        end
        Y, logdet = forward(X, L)
        f = log_likelihood(Y) - logdet
        if with_grad
            ΔY = ∇log_likelihood(Y)
            if backward_y
                ΔX = backward(ΔY, Y, L)[1]
            else
                ΔX = backward(ΔY, X, L)
            end
            return f, ΔX
        end
        return f
    end

    # Gradient test w.r.t. input X
    println("    $name: testing input")
    loss_test = function (X; with_grad=false)
        return loss(nothing, X; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(X); with_grad=true)

    if isnothing(do_flux)
        println("           $name input: Starting Flux forward")
        f_f, back, ΔX_f, do_flux = try 
            f_f, back = Flux.pullback(loss_test, deepcopy(X))
            println("           $name input: Starting Flux backward")
            ΔX_f = back(1f0)[1]
            println("           $name input: Done Flux")
            f_f, back, ΔX_f, true
        catch
            nothing, nothing, nothing, false
        end
    elseif do_flux
        println("           $name input: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(X))
        println("           $name input: Starting Flux backward")
        ΔX_f = back(1f0)[1]
        println("           $name input: Done Flux")
    end

    println("    $name input: First with our gradient")
    grad_test(loss_test, deepcopy(X), deepcopy(dX), deepcopy(ΔX); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT)

    if do_flux
        @show norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f))
        println("    $name input: Then with Flux's gradient")
        grad_test(loss_test, deepcopy(X), deepcopy(dX), deepcopy(ΔX_f); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT)
        @test f_f ≈ f0
        @test norm(ΔX - ΔX_f) ./ max(norm(ΔX), norm(ΔX_f)) < tol
    end

    # Gradient test w.r.t. parameters
    clear_grad!(L)
    clear_grad!(P)
    loss_test = function (P; with_grad=false)
        return loss(P, X; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(P); with_grad=true)
    ΔP = deepcopy(get_grads(L))

    if do_flux
        println("           $name parameters: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(P))

        println("           $name parameters: Starting Flux backward")
        ΔP_fT = back(1f0)[1]
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        println("           $name parameters: Done Flux")

        flux_forward = function (P)
            set_params!(L_params, P)
            forward(X, L)
        end
        (Y, logdet), back_Y_logdet = Flux.pullback(flux_forward, deepcopy(P))
        f1, back_f1 = Flux.pullback(log_likelihood, Y)
        f, back_f = Flux.pullback((a,b) -> a - b, f1, logdet)

        Δf = 1f0
        Δf1, Δlogdet = back_f(Δf)
        ΔY, = back_f1(Δf1)
        ΔP1t, = back_Y_logdet((ΔY, Δlogdet))
        ΔP1 = [Parameter(a.data, a.grad) for a in ΔP1t]

        @test f_f ≈ f0
        @test norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f), 1) < tol
    end

    # error("done")
    set_params!(L, deepcopy(P))

    # Test each parameter.
    do_taylor_test = length(P) < 6
    for (i, (p, dp, Δp)) in enumerate(zip(P, dP, ΔP))
        println("    $name: testing parameter $i")
        loss_test = function (p_vec)
            P = deepcopy(P)
            P[i].data = p_vec
            return loss(P, X; with_grad=false)
        end

        if do_flux
            Δp_f = ΔP_f[i]
        end

        if do_taylor_test
            println("    $name parameter $i: First with our gradient")
            do_flux && @show norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f), 1)
            grad_test(loss_test, deepcopy(p.data), deepcopy(dp.data), deepcopy(Δp.data); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
        end

        if do_flux
            do_taylor_test && @show norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f), 1)
            @test norm(Δp - Δp_f) ./ max(norm(Δp), norm(Δp_f), 1) < tol skip=false
            do_taylor_test && println("    $name parameter $i: Then with Flux's gradient")
            do_taylor_test && grad_test(loss_test, deepcopy(p.data), deepcopy(dp.data), deepcopy(Δp_f.data); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
        end
    end

    # Test all parameters.

    # Gradient test w.r.t. parameters
    println("   $name: testing all parameters")
    clear_grad!(L)
    clear_grad!(P)
    loss_test = function (P; with_grad=false)
        return loss(P, X; with_grad)
    end
    f0, ΔX = loss_test(deepcopy(P); with_grad=true)
    ΔP = deepcopy(get_grads(L))

    if do_flux
        println("           $name parameters: Starting Flux forward")
        f_f, back = Flux.pullback(loss_test, deepcopy(P))

        println("           $name parameters: Starting Flux backward")
        ΔP_fT = back(1f0)[1]
        ΔP_f = [Parameter(a.data, a.grad) for a in ΔP_fT]

        println("           $name parameters: Done Flux")

        @test norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f), 1) < tol
    end

    println("    $name all parameter: First with our gradient")
    if do_flux
        @show norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f), 1)
    end
    grad_test(loss_test, deepcopy(P), deepcopy(dP), deepcopy(ΔP); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)

    if do_flux
        @show norm(ΔP - ΔP_f) ./ max(norm(ΔP), norm(ΔP_f), 1)
        println("    $name all parameter: Then with Flux's gradient")
        grad_test(loss_test, deepcopy(P), deepcopy(dP), deepcopy(ΔP_f); maxiter=20, h0=4f0, hfactor=5f-1, eT=TT, unittest=:test)
    end
end
