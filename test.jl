##### Load dependencies and network data #####
using wq_robust_mpc
using Revise


net_name = "Net1" # "Threenode", "Net1", "Net3"
network = load_network(net_name)






##### Simulation functions #####

# EPANET solver
sim_type = "hydraulic" # "hydraulic", "chlorine", "age``, "trace"
sim_days = 7
n_t = get_hydraulic_time_steps(network, net_name, sim_days)
sim_results = epanet_solver(network, sim_type; sim_days=sim_days)

# Water quality solver
Δt = 60
source_cl = repeat([1.5], network.n_r)
u = ones(1, n_t)
x = wq_solver(network, sim_days, Δt, source_cl, u; kb=0.3, kw=0.05, disc_method="explicit-central")





##### Plotting functions #####

# network layout
plot_network_layout(network; pumps=true, legend=true, legend_pos=:lc, fig_size=(600, 450), save_fig=true)

# simulation results
state_to_plot = "pressure" # "pressure", "head", "demand", "flow", "flowdir", "velocity", "chlorine", "age", "trace"
state_df = getfield(sim_results, Symbol(state_to_plot))

time_to_plot = 1
plot_network_sim(network, state_df, state_to_plot; time_to_plot=time_to_plot+1, fig_size=(600, 450), pumps=true, save_fig=true)  # time offset since simulation starts at t = 0

elements_to_plot = network.node_names[end] # e.g., network.node_names[1:4] or network.link_names[1:4]
plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(700, 350), save_fig=true)

# EPANET v. water quality solver
