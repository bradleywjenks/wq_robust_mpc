##### Load dependencies and network data #####
using wq_robust_mpc
using Revise
using Plots


net_name = "Threenode" # "Threenode", "Net1", "Net3"
network = load_network(net_name)






##### Simulation functions #####

# solver inputs
sim_days = 1
Δk = 3600 # hydraulic time step (default is Δk = 3600 seconds)
Δt = 5 # water quality time step (default is Δt = 60 seconds)
kb = 0 # (1/day)
kw = 0 # (m/day)
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind", "explicit-upwind", "implicit-central"
source_cl = repeat([0.5], network.n_r)
control_pattern = "constant" # "constant", "random", "user-defined"
b_loc, b_u = get_booster_inputs(network, net_name, sim_days, Δk, Δt; control_pattern=control_pattern) # booster control locations and settings
x0 = 0.5 # initial conditions

# EPANET solver
sim_type = "chlorine" # "hydraulic", "chlorine", "age``, "trace"
sim_results = epanet_solver(network, sim_type; sim_days=sim_days, source_cl=source_cl, Δt=Δt, Δk=Δk, x0=x0, kb=kb, kw=kw)

# Water quality solver
x = wq_solver(network, sim_days, Δt, source_cl; kb=kb, kw=kw, disc_method=disc_method, Δk=Δk, x0=x0) # without booster control
x = wq_solver(network, sim_days, Δt, source_cl; kb=kb, kw=kw, disc_method=disc_method, Δk=Δk, x0=x0, b_loc=b_loc, b_u=b_u) # with booster control

plot(x[3, :])





##### Plotting functions #####

# network layout
plot_network_layout(network; pumps=true, legend=true, legend_pos=:lc, fig_size=(600, 450), save_fig=true)

# simulation results
state_to_plot = "chlorine" # "pressure", "head", "demand", "flow", "flowdir", "velocity", "chlorine", "age", "trace"
state_df = getfield(sim_results, Symbol(state_to_plot))

time_to_plot = 1
plot_network_sim(network, state_df, state_to_plot; time_to_plot=time_to_plot+1, fig_size=(600, 450), pumps=true, save_fig=true)  # time offset since simulation starts at t = 0

elements_to_plot = network.node_names[7] # e.g., network.node_names[1:4] or network.link_names[1:4]
plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(700, 350), save_fig=true)

# EPANET v. water quality solver





