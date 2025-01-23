# ================================================================================
# BLP Implementation
# Alex Lan, Jan 22 2025
# ================================================================================

using LinearAlgebra, DataFrames, Statistics, Distributions
using Optim, Plots, CSV

ps_dir = dirname(dirname(@__DIR__));
data_dir = joinpath(joinpath(ps_dir, "data"), "ps1_BLP");
res_dir = joinpath(ps_dir, "results");
df = CSV.File(joinpath(data_dir, "ps1_ex4.csv"))|> DataFrame;

# setup
J = 6;
T = 100;
L = 40;
X = [df.p df.x];
s = df.shares;
Z = Matrix(df[:, [:z1, :z2, :z3, :z4, :z5, :z6]]);
mu = [0.0, 0.0];
v_dist = MvNormal(mu, I(2));
tol = 1e-14;
max_iter = 1e4;

gamma11 = collect(-5:0.125:-4);
gamma21 = collect(-1:0.5:1);
gamma22 = collect(-1:0.5:1);

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

function gmm(δ, step = 1)

    # step 1
    W = I(size(Z, 2));
    step1 = optimize( β -> gmm_obj(β, δ, X, Z, W), [1.0; 1.0], BFGS());
    β1 = Optim.minimizer(step1);
    obj1 = gmm_obj(β1, δ, X, Z, W)
    if step == 1
        return β1, obj1
    end

    # the optimal weighting matrix
    res1 = δ .- X * β1;
    S = (Z' * res1) * (res1' * Z) ./ size(Z, 1);
    W_new = inv(S); 
    W_new = W_new ./ maximum(abs.(W_new)) + 1e-12*I(J);

    # step 2
    step2 = optimize( β -> gmm_obj(β, δ, X, Z, W_new), β1, BFGS());
    β2 = Optim.minimizer(step2);
    obj2 = gmm_obj(β2, δ, X, Z, W_new)
    if step == 2
        return β2, obj2
    end

end

### Implementation

v_vec = hcat([rand(v_dist) for i in 1:T*L]...);

results = []
for γ11 in gamma11, γ21 in gamma21, γ22 in gamma22
    Γ = [γ11 0; γ21 γ22];
    δ = get_delta(Γ, v_vec);
    β, obj = gmm(δ, 1);
    result = [obj, γ11, γ21, γ22, β]
    push!(results, result);
end

res_check = [val[1] for val in results];
sol = results[argmin(res_check)];
sorted_results = sort(results, by = x -> x[1])



## test
# Γ = [-4.5 0; 0 0];
# δ = get_delta(Γ, v_vec);
# β, obj = gmm(δ, 1)

# W = I(size(Z, 2));
# step1 = optimize( β -> gmm_obj(β, δ, X, Z, W),
#         [1.0; 1.0], LBFGS());
# β1 = Optim.minimizer(step1);

# # the optimal weighting matrix
# res1 = δ .- X * β1;

# # Z = Z ./ maximum(abs.(Z), dims=1);
# # res1 = res1 ./ maximum(abs.(res1));
# S = (Z' * res1) * (res1' * Z) ./ size(Z, 1);
# # S = S ./ maximum(abs.(S))
# W_new = inv(S);
# W_new = W_new ./ maximum(abs.(W_new)) + 1e-12*I(J)

# step2 = optimize( β -> gmm_obj(β, δ, X, Z, W_new),
#         β1, LBFGS());
# β2 = Optim.minimizer(step2);

# obj = gmm_obj(β2, δ, X, Z, W_new)