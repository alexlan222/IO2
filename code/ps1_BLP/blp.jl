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
Z = hcat(Matrix(df[:, [:z1, :z2, :z3, :z4, :z5, :z6, :x]]), ones(J*T));
mu = [0.0, 0.0];
v_dist = MvNormal(mu, I(2));
tol = 1e-14;
max_iter = 1e5;

# u = (-α + γ11v1)p +(β + γ21v1 + γ22v2)x + ξ

v_vec = hcat([rand(v_dist) for i in 1:T*L]...);

function blp_objective_nonlinear(γ, X, s, Z, v_vec)

    Γ = [γ[1] 0.0; γ[2] γ[3]]
    δ = get_delta(Γ, X, s, v_vec)
    β1, obj1 = gmm(δ, X, Z)
    return obj1
end

function obj_fn(param)
    blp_objective_nonlinear(param, X, s, Z, v_vec)
end


res = optimize(obj_fn, [-1.0, 0.0, 1.0], BFGS(),
                   Optim.Options(show_trace=true, store_trace=true, iterations=50))

   
γ = Optim.minimizer(res)
Γ = [γ[1] 0.0; γ[2] γ[3]]
δ = get_delta(Γ, X, s, v_vec)
β, obj = gmm(δ, X, Z)


eps_deep = fill(0., J, J, T);
for t in 1:T
    Xt = X[(J*(t-1)+1) : J*t, :];
    vt = v_vec[:, (L*(t-1)+1) : L*t]
    δt = δ[(J*(t-1)+1) : J*t]
    eps_t = get_eps_t(β, Xt, Γ, vt, δt);
    eps_deep[:,:,t] = eps_t;
end

eps = dropdims(mean(eps_deep, dims = 3), dims = 3);

p = dropdims(mean(reshape(df.p, J, :), dims = 2), dims = 2);
s = dropdims(mean(reshape(df.shares, J, :), dims = 2), dims = 2);
x = dropdims(mean(reshape(df.x, J, :), dims = 2), dims = 2);
dis = DataFrame(round.(hcat(p, s, x), digits = 3), [:price, :share, :quality]); 
