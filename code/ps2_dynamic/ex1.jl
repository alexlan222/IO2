# ================================================================================
# Exercise 1
# Alex Lan, Feb 17 2025
# ================================================================================

include("ddc_functions.jl")

# Q1 graphs
plot_ccp1(ddc(θ1 = -1, θ2 = -10, β = 0.95), "ccp_10_95")
plot_ccp1(ddc(θ1 = -1, θ2 = -20, β = 0.95), "ccp_20_95")
plot_ccp1(ddc(θ1 = -1, θ2 = -10, β = 0.5), "ccp_10_50")
plot_ccp1(ddc(θ1 = -1, θ2 = -20, β = 0.5), "ccp_20_50")

# Q2 functions 


# Q3 simulate and estimate

ddc1 = ddc(θ1 = -1, θ2 = -10)
data = forward_sim(ddc1, 10000)
θ_opt = two_step_ccp(data, ddc1, 1e-6) # sensitive to the choice of ε

plot_ccp1(ddc(θ1 = -1, θ2 = -10), "true_ccp_q3")
plot_ccp1(ddc(θ1 = θ_opt[1], θ2 = θ_opt[2]), "implied_ccp_q3")