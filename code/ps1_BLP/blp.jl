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
L = 50;
X = [df.p df.x];
s = df.shares;
Z = Matrix(df[:, [:z1, :z2, :z3, :z4, :z5, :z6, :x]]);
mu = [0.0, 0.0];
v_dist = MvNormal(mu, I(2));
tol = 1e-14;
max_iter = 1e5;

gamma11 = collect(4:1:10);
gamma21 = collect(-0.4:0.2:0.4);
gamma22 = collect(-0.4:0.2:0.4);

### Implementation

v_vec = hcat([rand(v_dist) for i in 1:T*L]...);

results = []
for γ11 in gamma11, γ21 in gamma21, γ22 in gamma22
    Γ = [γ11 0; γ21 γ22];
    δ = get_delta(Γ, v_vec);
    β, obj = gmm(δ, X, Z, [-1.0; 1.0], 1);
    result = [obj, γ11, γ21, γ22, β, δ];
    push!(results, result);
end

sorted_results = sort(results, by = x -> x[1])
sol = sorted_results[1];

# 8.14818175067673, 7, -0.2, 0.0, [-0.9048011815751427, -2.954760690482228]
Γ = [sol[2] 0; sol[3] sol[4]];
β = sol[5];
δ = sol[6];

eps_deep = fill(0., J, J, T);
for t in 1:T
    Xt = X[(J*(t-1)+1) : J*t, :];
    vt = v_vec[:, (L*(t-1)+1) : L*t]
    eps_t = get_eps_t(β, Xt, Γ, vt, δ);
    eps_deep[:,:,t] = eps_t;
end

eps = dropdims(mean(eps_deep, dims = 3), dims = 3);
