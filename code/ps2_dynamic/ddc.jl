# ================================================================================
# Dynamic Discrete Choice
# Alex Lan, Feb 14 2025
# ================================================================================

using LinearAlgebra, DataFrames, Statistics, Distributions
using Parameters, Random, Optim, Plots, CSV
using Base.Threads 

repo_dir = dirname(dirname(@__DIR__));
res_dir = joinpath(joinpath(repo_dir, "results"), "ps2");

@with_kw struct ddc
    θ1 = -0.2;
    θ2 = -10;
    X = 30;
    x = float(collect(0:(X-1)));
    f = [0.3, 0.6]
    N = length(x);
    A = 2;
    β = 0.95
end

function utility(x, θ) 
    u = zeros(length(x), 2)
    u[:,1] .= θ[1] * x;
    u[:,2] .= θ[2];
    return u
end

function get_T(x, f)
    N = length(x);
    T = zeros(N, N);
    for i in 1:N
        if i == N -1
            T[i, i] = f[1];
            T[i, i+1] = 1 - f[1];
        elseif i == N
            T[i, i] = 1;
        else
            T[i, i] = f[1];
            T[i, i+1] = f[2];
            T[i, i+2] = 1 - f[1] -f[2];
        end
    end
    T2 = hcat(ones(N), zeros(N, N-1));
    T3 = cat(T, T2, dims=3);
    return T3
end

function get_w(u, T, β)
    max_iter = 1000;
    tol = 1e-6;
    iter = 0;
    sup_norm = 999;
    N, A = size(u)
    w_old  = zeros(N);
    w_new = zeros(N);
    while iter < max_iter && sup_norm > tol
        iter += 1;
        w_mat = repeat(w_old, 1, A);
        w_a_mat = reduce(hcat,[u[:,i] .+ β .* T[:,:,i]*w_old for i in 1:A]);

        # avoid numerical instability
        w_max = maximum(w_a_mat);
        w_a_mat = w_a_mat .- w_max;
        w_new .= w_max .+ log.(sum(exp.(w_a_mat), dims=2));

        sup_norm = maximum(abs.(w_new .- w_old));
        w_old = w_new;
    end
    return w_new
end

function vc(u, T, w, β)
    vc = reduce(hcat,[u[:,i] .+ β .* T[:,:,i]*w for i in 1:A]);
    return vc
end

function get_ccp(vc)
    vc_max = maximum(vc);
    vc = vc .- vc_max;
    ccp = exp.(vc) ./ sum(exp.(vc), dims=2);
    return ccp  
end

function plot_ccp1(ddc, θ2, β, fig)
    u = utility(ddc.x, ddc.θ1, θ2);
    T = get_T(ddc.x, ddc.f);
    w = get_w(u, T, β);
    ccp = get_ccp(vc(u, T, w, β));

    p = plot(ccp[:,1], title = string("CCP (θ2=",θ2,", β=",β,")"),
        xlabel = "state", ylabel = "maintenance", legend = false)
    savefig(p, joinpath(res_dir, fig * ".png"))
end

# Q1 graphs
ddc1 = ddc()
plot_ccp1(ddc1, -10, 0.95, "ccp_10_95")
plot_ccp1(ddc1, -5, 0.95, "ccp_5_95")
plot_ccp1(ddc1, -10, 0.5, "ccp_10_50")
plot_ccp1(ddc1, -5, 0.5, "ccp_5_50")

# model inversion
function vhatb_from_ccp(ddc, phat, θ)
    # phat is N*2 matrix of CCP
    vhat = log.(phat[:,1] ./ phat[:,ddc.A]);
    vhat = hcat(vhat, zeros(ddc.N));
    vmax = maximum(vhat);
    vhat = vhat .- vmax;
    log_sum_exp = vmax .+ log.(sum(exp.(vhat),dims = 2))

    T = get_T(ddc.x, ddc.f);
    u = utility(ddc.x, θ[1], θ[2]);
    β = ddc.β; 
    A = ddc.A
    vhatb = reduce(hcat, [u[:,i] .- u[:,A] .+ 
                        β .* (T[:,:,i] * log_sum_exp .- 
                             T[:,:,A] * log_sum_exp) for i in 1:A])
    
    return vhatb
end

function likelihood_two_step_ccp(θ, data, ddc, phat)
    vhatb = vhatb_from_ccp(ddc, phat, θ);
    id_lis = data.x
    vhatb_dt = vhatb[id_lis, :]
    LL = sum(data .* log.(vhatb_dt))
    return LL
end

# draw simulations
Tsim = 100;
function forward_sim(ddc, Tsim)
    uf = rand(Uniform(0, 1), Tsim);
    us = rand(Gumbel(-MathConstants.γ, 1), (Tsim, 2));
    θ = [-0.2, -10];
    u = utility(ddc.x, θ);
    T = get_T(ddc.x, ddc.f);
    β = ddc.β;
    w = get_w(u, T, β);
    v_c = vc(u, T, w, β);
    x_init_id = 1;
    x_id_sim = [];
    a_id_sim = [];
    x_old_id = x_init_id
    for t in 1:Tsim
        vc_t = v_c[x_old_id, :] .+ us[t,:];
        a = argmax(vc_t);
        cumsum_pr = cumsum(T[x_old_id, :, a]);
        x_new_id = findfirst(cumsum_pr .>= uf[t]);
        push!(x_id_sim, x_new_id)
        push!(a_id_sim, a)
        x_old_id = x_new_id;
    end
end
