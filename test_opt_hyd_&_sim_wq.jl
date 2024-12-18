##### Baseline scenarion 1: optimize hydraulics, run water quality simulation without booster station optimization #####

##### Load dependencies and network data, and optimization parameters #####
using wq_robust_mpc
using Revise
using Plots


# load network data
net_name = "Net3" # "Threenode", "Net1", "Net3", "2loopsNet", "Net25", "modena", "BWFLnet", "L-town"
network = load_network(net_name);

# create optimization parameters
sim_days = 1
Δk = 60 * 60
Δt = 60 * 60
QA = true
pmin = 5
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
obj_type = "cost" # "cost" # "AZP"
x_wq_bounds = (0.2, 4)
u_wq_bounds = (0, 5)
x_wq_0 = 0.5 # initial water quality conditions
J = nothing
opt_params = make_prob_data(network, Δt, Δk, sim_days, disc_method; pmin=pmin, QA=QA, x_wq_bounds=x_wq_bounds, u_wq_bounds=u_wq_bounds, obj_type=obj_type, x_wq_0=x_wq_0, J=J);

# run optimization solver
cpu_time = @elapsed begin
    solver = "Gurobi" # "Gurobi", "Ipopt"
    heuristic = false
    integer = true
    warm_start = false
    opt_results = optimize_hydraulic_wq(network, opt_params; x_wq_0=x_wq_0, solver=solver, integer=integer, warm_start=warm_start, heuristic=heuristic)
end 

h_tk = opt_results.h_tk;
q = opt_results.q;

# Plot optimization results
#= elements_to_plot = 1; #network.node_names[network.pump_idx]
state_df = getfield(opt_results, :h_tk) #Symbol(state_to_plot))
plot_timeseries_sim(network, state_df, "tank head", elements_to_plot; fig_size=(700, 350), save_fig=true) =#

#=
##### store optimal schedule/control settings and use for wq simulation #####
##### wq simulation #####

# solver inputs
sim_days = 1
Δk = 60 * 60 # hydraulic time step (default is Δk = 3600 seconds)
Δt = 60 * 5  # water quality time step (default is Δt = 60 seconds)
kb = 0.5 # (1/day)
kw = 0 # (m/day)
disc_method = "implicit-upwind" # "implicit-upwind", "explicit-central", "explicit-upwind"
source_cl = repeat([0.5], network.n_r)
control_pattern = "constant" # control pattern of booster stations: "constant", "random", "user-defined"
b_loc, b_u = get_booster_inputs(network, net_name, sim_days, Δk, Δt; control_pattern=control_pattern) # booster control locations and settings (flow paced booster)
x0 = 0 # initial conditions

# EPANET solver
sim_type = "chlorine" # "hydraulic", "chlorine", "age``, "trace"
sim_results = epanet_solver(network, sim_type; sim_days=sim_days, source_cl=source_cl, Δt=Δt, Δk=Δk, x0=x0, kb=kb, kw=kw, b_loc=b_loc, b_u=b_u)

# Water quality solver (from scracth)
#=x = wq_solver(network, sim_days, Δt, Δk, source_cl, disc_method; kb=kb, kw=kw, x0=x0)
cpu_time = @elapsed begin
    x = wq_solver(network, sim_days, Δt, Δk, source_cl, disc_method; kb=kb, kw=kw, x0=x0, b_loc=b_loc, b_u=b_u) # with booster control
end=#
=#


##### Plotting functions #####

# network layout
plot_network_layout(network; pumps=true, legend=true, legend_pos=:lc, fig_size=(600, 450), save_fig=true)
plot_timeseries_opt(network, opt_results, "pumps"; fig_size=(600, 450), save_fig=false)
plot_timeseries_opt(network, opt_results, "tanks"; fig_size=(600, 450), save_fig=false)

#=
# EPANET simulation results
state_to_plot = "chlorine" # "pressure", "head", "demand", "flow", "flowdir", "velocity", "chlorine", "age", "trace"
state_df = getfield(sim_results, Symbol(state_to_plot))

# EPANET plotting only
time_to_plot = 100
plot_network_sim(network, state_df, state_to_plot; time_to_plot=time_to_plot+1, fig_size=(600, 450), pumps=true, save_fig=true)  # time offset since simulation starts at t = 0
elements_to_plot = network.node_names[end] # e.g., network.node_names[1:4] or network.link_names[1:4]
plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(700, 350), save_fig=true) =#
