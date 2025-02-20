# ================================================================================
# Exercise 3
# Alex Lan, Feb 20 2025
# ================================================================================

using Distributions, DataFrames, CSV

repo_dir = dirname(dirname(@__DIR__));
data_dir = joinpath(joinpath(repo_dir, "data"), "ps2_dynamic");

df = CSV.File(joinpath(data_dir, "ps2_ex3.csv"))|> DataFrame;
dϵ = Normal(0,1)

function LL_obj_ex3(θ, df)
    cdf_N =  @. cdf(dϵ, -θ[1]* df.x + θ[2] + θ[3]*log(df.n + 1)) - 
            cdf(dϵ, -θ[1]* df.x + θ[2] + θ[3]*log(df.n))
    LL = sum(log.(cdf_N))
    return -LL
end

θ_init = [1., 5., 1.];
result = optimize(θ -> LL_obj_ex3(θ, df), θ_init,
    BFGS(), Optim.Options(show_trace = true));
θ_opt = Optim.minimizer(result);
# 2.152706423074553
# 1.5712495267996167
# 10.602362939204927