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
L = 1000;
X = [df.p df.x];
s = df.shares;
Z = hcat(Matrix(df[:, [:z1, :z2, :z3, :z4, :z5, :z6, :x]]), fill(1., T*J));
mu = [0.0, 0.0];
v_dist = MvNormal(mu, I(2));
tol = 1e-14;
max_iter = 1e5;

gamma11 = collect(2);
gamma21 = collect(-0.01:0.01:0.01);
gamma22 = collect(-0.03:0.01:-0.01);

# u = (-α + γ11v1)p +(β + γ21v1 + γ22v2)x + ξ

v_vec = hcat([rand(v_dist) for i in 1:T*L]...);

results = []
for γ11 in gamma11, γ21 in gamma21, γ22 in gamma22
    Γ = [γ11 0; γ21 γ22];
    δ = get_delta(Γ, v_vec);
    β, obj = gmm(δ, X, Z, [-1.0; 1.0], 2);
    result = [obj, γ11, γ21, γ22, β, δ];
    push!(results, result);
end

sorted_results = sort(results, by = x -> x[1]);
sol = sorted_results[1]

# 24.918809058952128, 7, -0.01, -0.01, Any[-3.1020556581862415, 0.7600865529273264]

Γ = [sol[2] 0; sol[3] sol[4]];
β = sol[5];
δ = sol[6];

eps_deep = fill(0., J, J, T);
for t in 1:T
    Xt = X[(J*(t-1)+1) : J*t, :];
    vt = v_vec[:, (L*(t-1)+1) : L*t]
    δt = δ[(J*(t-1)+1) : J*t]
    eps_t = get_eps_t(β, Xt, Γ, vt, δt);
    eps_deep[:,:,t] = eps_t;
end

eps = dropdims(mean(eps_deep, dims = 3), dims = 3);


#### testing
Γ = [3 0; -1 0.0];
δ = get_delta(Γ, v_vec);
W = I(size(Z, 2));
X_cons = hcat(X, fill(1., size(X,1)));
step1 = optimize( β -> gmm_obj(β, δ, X_cons, Z, W), [-1.0; 1.0; 1.0], BFGS());

β1 = Optim.minimizer(step1);
obj1 = gmm_obj(β1, δ, X_cons, Z, W)

res1 = δ .- X_cons * β1;
S = (Z' * res1) * (Z' * res1)' ./ size(Z, 1);
S = S .+  1e-12*I(size(S, 2)); 
W_new = inv(S);
step2 = optimize( β -> gmm_obj(β, δ, X_cons, Z, W_new), β1, BFGS());
β2 = Optim.minimizer(step2);
obj2 = gmm_obj(β2, δ, X_cons, Z, W_new);

X̂=(Z*inv(Z'*Z)*Z'*X);
β1 = (inv(X̂'*X̂)*X̂'*δ);

res1 = δ .- X * β1;
res1 = (res1 .- mean(res1)) ./ std(res1);
S = (Z' * res1) * (Z' * res1)' ./ size(Z, 1) .+ 1e-8 *I(size(Z, 2));
W_new = pinv(S); 

