# ================================================================================
# homogeneous logit demand
# Alex Lan, Jan 22 2025
# ================================================================================

include("blp_functions.jl")

ps_dir = dirname(dirname(@__DIR__));
data_dir = joinpath(joinpath(ps_dir, "data"), "ps1_BLP");
res_dir = joinpath(ps_dir, "results");
df = CSV.File(joinpath(data_dir, "ps1_ex2.csv"))|> DataFrame;

T = 1000;
J = 6;

X = [df.Prices df.x];
Z = [df.z df.x];

s0_vec = [];
for t in 1:T
    s0 = 1 - sum(df[df.market .== t, :Shares]);
    append!(s0_vec, s0)
end

df = leftjoin(df, DataFrame(market = 1:T,s0 = s0_vec), on = :market);
y = log.(df.Shares) .- log.(df.s0);
β, obj = gmm(y, X, Z, [1.0; 1.0], 1);

eps_deep = fill(0., J, J, T);
for t in 1:T
    Xt = X[(J*(t-1)+1) : J*t, :];
    eps_t = get_eps_t(β, Xt, 0);
    eps_deep[:,:,t] = eps_t;
end

eps = dropdims(mean(eps_deep, dims = 3), dims = 3);



