##### Test functions under development... #####
using wq_robust_mpc
using Revise

net_name = "Net1" # "threenode", "Net1", "Net3"
network = load_network(net_name)

# network plotting
plot_network_layout(network; pumps=true, legend=true, legend_pos=:lc, fig_size=(600, 450))

# EPANET simulation
sim_type = "hydraulic"
sim_results = epanet_solver(network, sim_type)

# simulation results plotting
state_to_plot = "velocity" # "pressure", "head", "demand", "flow", "velocity", "chlorine", "age", "trace"
state_df = getfield(sim_results, Symbol(state_to_plot))

time_to_plot = 4
plot_network_sim(network, state_df, state_to_plot; time_to_plot=time_to_plot+1, fig_size=(600, 450), pumps=true)  # time offset since simulation starts at t = 0

elements_to_plot = network.node_names[1:4] # e.g., network.node_names[1:4] or network.link_names[1:4]
plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(700, 350))