##### Load dependencies and network data #####
using wq_robust_mpc
using Revise
using Plots


net_name = "Net1" # "Threenode", "Net1", "Net3"
network = load_network(net_name)


##### Optimization functions #####

# optimization inputs
sim_days = 1
Δk = 60 * 60
Δt = 60
kb = 0 # (1/day)
kw = 0 # (m/day)
disc_method = "explicit-central" # "explicit-central", "implicit-upwind"
source_cl = repeat([0.5], network.n_r)
b_loc, _ = get_booster_inputs(network, net_name, sim_days, Δk, Δt) # booster control locations
x0 = 0 # initial conditions
x_bounds = (0.5, 3)
u_bounds = (0, 5)


# optimize water quality
x, u = optimize_wq(network, sim_days, Δt, Δk, source_cl, b_loc, x0; kb=kb, kw=kw, disc_method=disc_method, x_bounds=x_bounds, u_bounds=u_bounds)
