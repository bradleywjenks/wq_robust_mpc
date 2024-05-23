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
Δt = 60 * 60
kb = 0.5 # (1/day)
kw = 0 # (m/day)
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
source_cl = repeat([0.5], network.n_r)
b_loc, _ = get_booster_inputs(network, net_name, sim_days, Δk, Δt) # booster control locations
x0 = 0 # initial conditions
x_bounds = (0, 4)
u_bounds = (0, 5)

# optimize water quality
c_r, c_j, c_tk, c_m, c_v, c_p, u = optimize_wq(network, sim_days, Δt, Δk, source_cl, b_loc, x0; kb=kb, kw=kw, disc_method=disc_method, x_bounds=x_bounds, u_bounds=u_bounds)

# plot optimization results
plot(c_tk[1, 1:end-1])
plot(c_j[5, 1:end-1])
plot(c_p[8, 1:end-1])
plot(c_m[1, 1:end-1])
plot(u[1, 1:end-1])

# plot simulation results
b_u = u[1, :]
Δk = 60 * 60
Δt = 60 * 5
sim_type = "chlorine" # "hydraulic", "chlorine", "age``, "trace"
sim_results = epanet_solver(network, sim_type; sim_days=sim_days, source_cl=source_cl, Δt=Δt, Δk=Δk, x0=x0, kb=kb, kw=kw, b_loc=b_loc, b_u=b_u)

node_to_plot = network.node_names[3]
state_to_plot = "chlorine"
state_df = getfield(sim_results, Symbol(state_to_plot))

plot(state_df[!, 1], state_df[!, string(node_to_plot)])
plot!(c_j[3, 1:end-1])
plot!(c_tk[1, :])
