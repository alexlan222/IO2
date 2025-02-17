# ================================================================================
# Exercise 2
# Alex Lan, Feb 17 2025
# ================================================================================

using Distributions, StatsPlots, Printf
using DataFrames, CSV

repo_dir = dirname(dirname(@__DIR__));
data_dir = joinpath(joinpath(repo_dir, "data"), "ps2_dynamic");
res_dir = joinpath(joinpath(repo_dir, "results"), "ps2");

df = CSV.File(joinpath(data_dir, "ps2_ex2.csv"))|> DataFrame;
df.milage_l = circshift(df.milage, 1);
df.rep = Int.((df.milage .- df.milage_l ).< 0);
df.a_id = circshift(df.rep, -1) .+ 1;
df[df.rep .== 1, :milage_l] .= 0;
df.g = df.milage .- df.milage_l;

allowmissing!(df, [:a_id, :g])
df[1,:g] = missing;
df[end,:a_id] = missing;

g = collect(skipmissing(df.g));
n_fit = fit(Normal, g)
p = histogram(g, normalize = true, alpha = 0.3, label="Data")
plot!(x -> pdf(n_fit, x), 0, maximum(g), 
    label=@sprintf("Normal (μ=%.2f, σ²=%.2f)", mean(n_fit), var(n_fit)))
savefig(p, joinpath(res_dir, "g_fit.png"))

K = 20;
minx = minimum(df.milage);
maxx = maximum(df.milage);
x_pts = LinRange(minx, maxx, K+1);
x_grid = vcat([[x_pts[i], x_pts[i+1]]' for i in 1:K]...);
delta = x_pts[2] - x_pts[1];

T1 = zeros(K, K)
for i in 1:K 
    if i == K 
        T1[i, K] = 1
    else
    T1[i, i] = cdf(n_fit, (1/2) * delta) 
    for j in (i+1):K 
        if j == K 
            T1[i,j] = 1 - cdf(n_fit, (j - i - 1 + (1/2)) * delta)
        else
            T1[i,j] = cdf(n_fit, (j - i + (1/2)) * delta) - cdf(n_fit, (j - i - 1 + (1/2)) * delta)
        end       
    end
    end
end

@assert all(isapprox.(sum(T1, dims=2), 1.0, atol=1e-10)) "Invalid transition probabilities"

T2 = [i == 1 ? cdf(n_fit, delta) : (cdf(n_fit, i*delta) - cdf(n_fit, (i-1)*delta)) for i in 1:K];
T2 = repeat(T2', K, 1)
@assert all(isapprox.(sum(T2, dims=2), 1.0, atol=1e-10)) "Invalid transition probabilities"
