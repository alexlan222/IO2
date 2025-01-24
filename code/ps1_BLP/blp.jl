# ================================================================================
# BLP Implementation
# Alex Lan, Jan 22 2025
# ================================================================================

include("blp_functions.jl")

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
Z = Matrix(df[:, [:z1, :z2, :z3, :z4, :z5, :z6, :x]]);
mu = [0.0, 0.0];
v_dist = MvNormal(mu, I(2));
tol = 1e-14;
max_iter = 1e5;

gamma11 = collect(1:0.25:4);
gamma21 = collect(-0.2:0.1:0.2);
gamma22 = collect(-0.2:0.1:0.2);

### Implementation

v_vec = hcat([rand(v_dist) for i in 1:T*L]...);

results = []
for γ11 in gamma11, γ21 in gamma21, γ22 in gamma22
    Γ = [γ11 0; γ21 γ22];
    δ = get_delta(Γ, v_vec);
    β, obj = gmm(δ, X, Z, [-1.0; 1.0], 1);
    result = [obj, γ11, γ21, γ22, β]
    push!(results, result);
end

res_check = [val[1] for val in results];
sol = results[argmin(res_check)];
sorted_results = sort(results, by = x -> x[1])

# 7.9977658341808135, 6, -0.5, 0.0, [-0.2894857727082014, -3.094971039967556]

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