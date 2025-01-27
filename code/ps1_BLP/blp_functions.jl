# ================================================================================
# BLP functions
# Alex Lan, Jan 22 2025
# ================================================================================

using LinearAlgebra, DataFrames, Statistics, Distributions
using Optim, Plots, CSV

function pr_mat(δ, X, Γ, v_vec)

    u_mat = δ .+ X * Γ * v_vec;
    u_max = maximum(u_mat); # avoid numerical overflow
    pr_mat = exp.(u_mat .- u_max) ./ (exp(-u_max) .+ sum(exp.(u_mat .- u_max), dims = 1));
    return pr_mat
end

function jacobian(s_mat)

    j_deep = [ t == k ? s_mat[t, i] - s_mat[t, i]^2 : -s_mat[t, i] * s_mat[k, i] 
        for t in 1:J, k in 1:J, i in 1:L ];
    j = dropdims(mean(j_deep, dims = 3), dims = 3);
    s_vec = mean(s_mat, dims = 2);
    j = j ./ s_vec
    return j
end

function delta_iter_t(X, s, Γ, v_vec)

    iter = 0
    sup_norm = 999
    δ = rand(J);
    converge_check = [];
    while iter < max_iter && sup_norm > tol
        iter += 1;
        s_mat = pr_mat(δ, X, Γ, v_vec);
        s_hat = mean(s_mat, dims = 2);
        sup_norm = maximum(abs.(s .- s_hat))
        push!(converge_check, sup_norm);
        # if iter > 1e3
        #     δ .= δ .+ inv(jacobian(s_mat))*(log.(s) .- log.(s_hat))
        # end
        δ .= δ .+ log.(s) .- log.(s_hat)
    end
    return δ, converge_check
end

function get_delta(Γ, v_vec)

    δ = []
    for t in 1:T
        Xt = X[(6*(t-1)+1) : 6*t, :];
        st = s[(6*(t-1)+1) : 6*t, :];
        δt, error = delta_iter_t(Xt, st, Γ, 
                v_vec[:, (L*(t-1)+1) : L*t]);
        append!(δ, δt)
    end
    return δ
end


### GMM (computation not numerically stable)

function gmm_obj(β, δ, X, Z, W)
    ξ = (δ .- X * β) ./ size(Z, 1);
    return (ξ' * Z) * W * (ξ' * Z)'
end

function gmm(δ, X, Z, β0, step = 1)

    # step 1
    W = I(size(Z, 2));
    step1 = optimize( β -> gmm_obj(β, δ, X, Z, W), β0, BFGS());
    β1 = Optim.minimizer(step1);
    obj1 = gmm_obj(β1, δ, X, Z, W)
    if step == 1
        return β1, obj1
    end

    # the optimal weighting matrix
    res1 = δ .- X * β1;
    S = (Z' * res1) * (res1' * Z) ./ size(Z, 1);
    W_new = inv(S); 
    W_new = W_new ./ maximum(abs.(W_new)) .+ 1e-12*I(size(Z, 2));

    # step 2
    step2 = optimize( β -> gmm_obj(β, δ, X, Z, W_new), β1, BFGS());
    β2 = Optim.minimizer(step2);
    obj2 = gmm_obj(β2, δ, X, Z, W_new)
    if step == 2
        return β2, obj2
    end
end

### Price elasticities

function get_eps_t(β, X, Γ, v_vec =[], δ=[])
    # X: J*2 matrix, β: 2*1
    if v_vec == []
        s_vec = pr_mat(δ, 0, 0, 0);
        eps = [ j == k ? β[1]*X[j,1]*(1 - s_vec[j]) : -β[1]*X[k,1]*s_vec[k] 
        for k in 1:J, j in 1:J];
    else
        s_mat = pr_mat(δ, X, Γ, v_vec);
        s_hat = mean(s_mat, dims = 2);
        a_vec = β[1] .+ Γ[1,1] .* v_vec[1, :]; 
        eps_deep = [j == k ? X[j,1]/s_hat[j]*a_vec[i]*(s_mat[j, i] - s_mat[j, i]^2) : 
        -X[k,1]/s_hat[j]*a_vec[i]*(s_mat[j, i] * s_mat[k, i]) for k in 1:J, j in 1:J, i in 1:L];
        eps = dropdims(mean(eps_deep, dims = 3), dims = 3);
    end
    return eps
end