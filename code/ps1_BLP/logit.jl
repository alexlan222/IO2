# ================================================================================
# homogeneous logit demand
# Alex Lan, Jan 22 2025
# ================================================================================

include("blp_functions.jl")
using NLsolve

ps_dir = dirname(dirname(@__DIR__));
data_dir = joinpath(joinpath(ps_dir, "data"), "ps1_BLP");
res_dir = joinpath(ps_dir, "results");
df = CSV.File(joinpath(data_dir, "ps1_ex2.csv"))|> DataFrame;

T = 1000;
J = 6;

X = [df.Prices df.x fill(1., T*J)];
Z = [df.z df.x fill(1., T*J)];

# generate the outside option shares
s0_vec = [];
for t in 1:T
    s0 = 1 - sum(df[df.market .== t, :Shares]);
    append!(s0_vec, s0)
end

df = leftjoin(df, DataFrame(market = 1:T,s0 = s0_vec), on = :market);

# 1. obtain an estimate for utility parameters
y = log.(df.Shares) .- log.(df.s0);
β, obj = gmm(y, X, Z, [1.0; 1.0], 1);
β_dis = round.(β, digits = 3);
#  -0.467 0.305

# 2. own and cross-product elasticities
eps_deep = fill(0., J, J, T);
for t in 1:T
    Xt = X[(J*(t-1)+1) : J*t, :];
    yt = y[(J*(t-1)+1) : J*t]
    eps_t = get_eps_t(β, Xt, 0, [], yt);
    eps_deep[:,:,t] = eps_t;
end

eps = dropdims(mean(eps_deep, dims = 3), dims = 3);
eps_dis = round.(eps, digits = 3);

# 6×6 Matrix{Float64}:
#  -1.25    0.321   0.321   0.321   0.321   0.321
#   0.323  -1.251   0.323   0.323   0.323   0.323
#   0.129   0.129  -1.289   0.129   0.129   0.129
#   0.127   0.127   0.127  -1.294   0.127   0.127
#   0.126   0.126   0.126   0.126  -1.291   0.126
#   0.129   0.129   0.129   0.129   0.129  -1.291

# 3. marginal cost
mc_deep = fill(0., J, T);

function get_mc(β, X, s)
    c = X[:,1] .+ (1 ./(β[1].*(1 .- s)))
    return c
end

for t in 1:T
    Xt = X[(J*(t-1)+1) : J*t, :];
    st = df.Shares[(J*(t-1)+1) : J*t]
    c_t = get_mc(β, Xt, st);
    mc_deep[:,t] = c_t;
end

mc = dropdims(mean(mc_deep, dims = 2), dims = 2);
p = dropdims(mean(reshape(df.Prices, 6, :), dims = 2), dims = 2);
s = dropdims(mean(reshape(df.Shares, 6, :), dims = 2), dims = 2);
mc_dis = DataFrame(round.(hcat(mc, p, s), digits = 3), [:mc, :price, :share]); 

#   │ mc       p        s       
# ──┼───────────────────────────
# 1 │   0.667    3.36     0.202
# 2 │   0.672    3.368    0.203
# 3 │   0.679    3.033    0.09
# 4 │   0.689    3.04     0.089
# 5 │   0.683    3.031    0.088
# 6 │   0.683    3.038    0.091


# 4. price and market share simulation when product 1 exits

xi = y .- X * β;
deleteat!(xi, 1:J:length(xi));
mc1 = mc_deep[2:J, :];
mc1 = vec(mc1);

tol = 1e-12;
max_iter = 1e5;

X1_ind = filter(x -> x ∉ collect(1:J:size(X,1)), collect(1:1:size(X,1)))
X1 = X[X1_ind, 2];

function share_pred(β, Xt, xit, p_old)
    exp_u = exp.(β[1] .* p_old .+ β[2] .* Xt .+ xit);
    share = exp_u ./ (1 + sum(exp_u));
    return share
end

function profit_foc!(p, mct, β, Xt, xit)
    p .- mct .+ 1 ./(β[1]*(1 .- share_pred(β, Xt, xit, p)))
end

sim_deep = fill(0., J-1, 2, T)
for t in 1:T
    Xt = X1[(t-1)*(J-1)+1 : t*(J-1)];
    xit = xi[(t-1)*(J-1)+1 : t*(J-1)];
    mct = mc1[(t-1)*(J-1)+1 : t*(J-1)];
    # p = p_cmt(β, Xt, xit, mct);
    p = nlsolve(p -> profit_foc!(p, mct, β, Xt, xit), fill(1., J-1)).zero
    share = share_pred(β, Xt, xit, p);
    sim_deep[:, 1, t] = p;
    sim_deep[:, 2, t] = share;
end

sim = dropdims(mean(sim_deep, dims = 3), dims = 3);
sim_dis = DataFrame(round.(sim, digits = 3), [:price, :share]); 
#      price    share   
# ──┼──────────────────
# 1 │   3.34     0.195
# 2 │   3.022    0.086
# 3 │   3.029    0.085
# 4 │   3.02     0.084
# 5 │   3.027    0.086

# 5. change in firms' profits and consumer welfare

p = copy(df.Prices);
deleteat!(p, 1:J:length(p));
s = copy(df.Shares);
deleteat!(s, 1:J:length(s));

sim_df =  vcat([sim_deep[:, :, i] for i in 1:size(sim_deep, 3)]...);
p1 = sim_df[:, 1];
s1 = sim_df[:, 2];
Δπ = (p1 .- mc1).*s1 .- (p .- mc1).*s;
Δπ_m = mean(reshape(Δπ, 5, 1000), dims = 2);
round.(Δπ_m, digits = 4)

# consumer welfare

y1 = β[1] .* p1 .+ β[2] .* X1 .+ xi;
ΔU = [];

function logsumexp(y)
    max_y = maximum(y)
    return max_y .+ log.(exp.(- max_y) + sum(exp.(y .- max_y)))
end

for t in 1:T
    Δu = logsumexp(y1[(t-1)*(J-1)+1 : t*(J-1)]) - logsumexp(y[(t-1)*J+1 : t*J]);
    append!(ΔU, Δu);
end

round.(mean(ΔU), digits = 4)

# function p_cmt(β, Xt, xit, mct)
#     p_old = fill(1., J-1);
#     iter = 0;
#     sup_norm = 999;
#     while iter < max_iter && sup_norm > tol
#         iter += 1;
#         p_new = mct .- 1 ./(β[1]*(1 .- share_pred(β, Xt, xit, p_old)));
#         sup_norm = maximum(abs.(p_new .- p_old));
#         p_old = p_new;
#     end
#     return p_new
# end