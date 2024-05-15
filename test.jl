##### Load dependencies and network data #####
using wq_robust_mpc
using Revise
using Plots


net_name = "Net1" # "Threenode", "Net1", "Net3"
network = load_network(net_name)






##### Simulation functions #####

# solver inputs
sim_days = 1
Δk = 3600
Δt = 60
kb = 0.5
kw = 0
disc_method = "explicit-central" # "explicit-central", "implicit-upwind"
source_cl = repeat([1.5], network.n_r)

# EPANET solver
sim_type = "hydraulic" # "hydraulic", "chlorine", "age``, "trace"
sim_results = epanet_solver(network, sim_type; sim_days=sim_days, source_cl=source_cl, Δt=Δt, Δk=Δk, kb=kb, kw=kw)
n_t = size(sim_results.timestamp)[1]
u = ones(1, n_t)

# Water quality solver
x, λ, s = wq_solver(network, sim_days, Δt, source_cl, u; kb=kb, kw=kw, disc_method=disc_method, Δk=Δk)

T_k = size(x, 2)
plot(x[network.n_r+1, 1:end])
plot(x[network.n_r+network.n_j+1, 1:1200])
plot(x[network.n_r+network.n_j+network.n_tk+1, 1:end])
plot(x[network.n_r+network.n_j+network.n_tk+network.n_m+network.n_v+609, 1:1000])
plot(x[end, 1:500])

plot(x[3, :])





##### Plotting functions #####

# network layout
plot_network_layout(network; pumps=true, legend=true, legend_pos=:lc, fig_size=(600, 450), save_fig=true)

# simulation results
state_to_plot = "flow" # "pressure", "head", "demand", "flow", "flowdir", "velocity", "chlorine", "age", "trace"
state_df = getfield(sim_results, Symbol(state_to_plot))

time_to_plot = 1
plot_network_sim(network, state_df, state_to_plot; time_to_plot=time_to_plot+1, fig_size=(600, 450), pumps=true, save_fig=true)  # time offset since simulation starts at t = 0

elements_to_plot = network.link_names[7] # e.g., network.node_names[1:4] or network.link_names[1:4]
plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(700, 350), save_fig=true)

# EPANET v. water quality solver





