##### Baseline scenario 1: optimize hydraulics, run water quality simulation without booster station optimization #####

##### Load dependencies and network data, and optimization parameters #####
using wq_robust_mpc
using Revise
using Plots
using CSV, DataFrames


# load network data
net_name = "Net1" # "Threenode", "Net1", "Net3", "2loopsNet", "Net25", "modena", "BWFLnet", "L-town"
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
max_pump_switch = 5

# define electricity tariff profile
electricity_tariff = "constant" # "user-defined"
if electricity_tariff == "constant"
    c_elec = ones(network.n_t,1)
else
    # "user-defined" tariffs (in GBP/kWh) are extracted from Nord Pool UK prices
    # see https://data.nordpoolgroup.com/auction/n2ex/prices?deliveryDate=2024-12-13&currency=GBP&aggregation=DeliveryPeriod&deliveryAreas=UK 
    # - tariff_1 corresponds to the 13.12.2024 
    tariff = "tariff_1" # "tariff_1", ...
    df = CSV.read(["electricity_tariffs/" * tariff * ".csv"], DataFrame)  # Load the CSV file into a DataFrame
    c_elec = df.hourly_electricity_tariff
end
opt_params = make_prob_data(network, Δt, Δk, sim_days, disc_method; pmin=pmin, QA=QA, x_wq_bounds=x_wq_bounds, u_wq_bounds=u_wq_bounds, obj_type=obj_type, x_wq_0=x_wq_0, J=J, max_pump_switch=max_pump_switch, c_elec=c_elec);

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


###### plotting functions #####
# network layout
fig1 = plot_network_layout(network; pumps=true, legend=true, legend_pos=:lc, fig_size=(600, 450), save_fig=false)
display(fig1)
# Plot optimization results
fig2 = plot_timeseries_opt(network, opt_results, "pumps"; fig_size=(600, 450), save_fig=false)
fig3 = plot_timeseries_opt(network, opt_results, "tanks"; fig_size=(600, 250), save_fig=false)
display(fig2)
display(fig3)


##### store optimal schedule/control settings and use for wq simulation #####
##### wq simulation #####

# solver inputs
sim_days = 1
Δk = 60 * 60 # hydraulic time step (default is Δk = 3600 seconds)
Δt = 60 * 15  # water quality time step (default is Δt = 60 seconds)
kb = 0.5 # (1/day)
kw = 0 # (m/day)
disc_method = "implicit-upwind" # "implicit-upwind", "explicit-central", "explicit-upwind"
source_cl = repeat([0.5], network.n_r)
control_pattern = "constant" # control pattern of booster stations: "constant", "random", "user-defined"
booster_control = false # true or false
if booster_control
    b_loc, b_u = get_booster_inputs(network, net_name, sim_days, Δk, Δt; control_pattern=control_pattern) # booster control locations and settings (pressure paced booster)
else
    b_loc = nothing
    b_u = nothing
end
# b_loc, b_u = get_booster_inputs(network, net_name, sim_days, Δk, Δt; control_pattern=control_pattern) # booster control locations and settings (flow paced booster)
x0 = 0.3 # initial conditions

# EPANET solver
sim_type = "chlorine" # "hydraulic", "chlorine", "age``, "trace"
# sim_results = epanet_solver(network, sim_type; sim_days=sim_days, source_cl=source_cl, Δt=Δt, Δk=Δk, x0=x0, kb=kb, kw=kw, b_loc=b_loc, b_u=b_u)

# Water quality solver (from scracth)
cpu_time = @elapsed begin
    wq_sim_results = wq_solver_fix_hyd(network, opt_results, opt_params, sim_days, Δt, Δk, source_cl, disc_method; kb=kb, kw=kw, x0=x0, b_loc=b_loc, b_u=b_u) 
end

node_to_plot = network.node_names[end-2]
fig4 = plot_wq_solver_comparison(network, [], wq_sim_results, node_to_plot, disc_method, Δt, Δk; fig_size=(700, 350), save_fig=false)
display(fig4)


##### Plotting functions #####
#=
# EPANET simulation results
state_to_plot = "chlorine" # "pressure", "head", "demand", "flow", "flowdir", "velocity", "chlorine", "age", "trace"
state_df = getfield(sim_results, Symbol(state_to_plot))

# EPANET plotting only
time_to_plot = 100
plot_network_sim(network, state_df, state_to_plot; time_to_plot=time_to_plot+1, fig_size=(600, 450), pumps=true, save_fig=true)  # time offset since simulation starts at t = 0
elements_to_plot = network.node_names[end] # e.g., network.node_names[1:4] or network.link_names[1:4]
plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(700, 350), save_fig=true) =#
