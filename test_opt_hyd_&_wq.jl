##### Load dependencies and network data, and optimization parameters #####
using wq_robust_mpc
using Revise
using Plots

# load network data
net_name = "Net1" # "Threenode", "Net1", "Net3", "2loopsNet", "Net25", "modena", "BWFLnet", "L-town"
network = load_network(net_name)

# create optimization parameters
sim_days = 1
Δk = 60 * 60 # Δk and Δt have to be equal
Δt = 60 * 60
QA = true
pmin = 10
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
obj_type = "AZP"
x_wq_bounds = (0, 4)
u_wq_bounds = (0, 5)
x_wq_0 = 0.5 # initial water quality conditions
J = nothing # manually set number of discretization points
opt_params = make_prob_data(network, Δt, Δk, sim_days, disc_method; pmin=pmin, QA=QA, x_wq_bounds=x_wq_bounds, u_wq_bounds=u_wq_bounds, obj_type=obj_type, x_wq_0=x_wq_0, J=J)


# run optimization solver
cpu_time = @elapsed begin
    solver = "Gurobi" # "Gurobi", "Ipopt"
    heuristic = false
    integer = true
    warm_start = false
    opt_results = optimize_hydraulic_wq(network, opt_params; x_wq_0=x_wq_0, solver=solver, integer=integer, warm_start=warm_start, heuristic=heuristic)
end

plot(opt_results.u_m[end, 1:end-1])

plot(opt_results.q⁺[1, 1:end-1])
plot!(opt_results.q⁻[1, 1:end-1])

plot(opt_results.h_tk[1, 1:end-1])

plot(opt_results.h_j[2, 1:end-1])