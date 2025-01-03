##### Load dependencies and network data #####
using wq_robust_mpc
using Revise
using Plots


net_name = "Net3" # "Threenode", "Net1", "Net3"
network = load_network(net_name)


##### Optimization functions #####

# optimization inputs
sim_days = 1
Δk = 60 * 60
Δt = 60 * 15
kb = 0.5 # (1/day)
kw = 0 # (m/day)
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
source_cl = repeat([0.5], network.n_r)
b_loc, _ = get_booster_inputs(network, net_name, sim_days, Δk, Δt) # booster control locations
x0 = 0.3 # initial conditions
x_bounds = (0.2, 4)
u_bounds = (0, 10)

# optimize water quality
c_r, c_j, c_tk, c_m, c_v, c_p, u = optimize_wq(network, sim_days, Δt, Δk, source_cl, b_loc, x0; kb=kb, kw=kw, disc_method=disc_method, x_bounds=x_bounds, u_bounds=u_bounds)

# plot optimization results
plot(c_tk[1, 1:end-1])
plot(c_j[end, 1:end-1])
plot(c_p[8, 1:end-1])
plot(c_m[1, 1:end-1])
plot(u[1, 1:end-1])


