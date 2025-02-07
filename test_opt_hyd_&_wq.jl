##### Load dependencies and network data, and optimization parameters #####
using wq_robust_mpc
using Revise
using Plots

# load network data
net_name = "Net1" # "Threenode", "Net1", "Net3", "2loopsNet", "Net25", "modena", "BWFLnet", "L-town"
network = load_network(net_name)

# create optimization parameters
sim_days = 1
Δk = 60 * 60 # Δk and Δt have to be equal (? what for?)
Δt = 60 * 60 # 15
kb = 0.5 # (1/day)
kw = 0 # (m/day)
QA = true
pmin = 0
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
source_cl = repeat([0.5], network.n_r) # or 0.25?
b_loc, _ = get_booster_inputs(network, net_name, sim_days, Δk, Δt) # booster control locations
#x0 = 0.5 # trying from Wang, Taha et al. (2021) + 1.0 at tank # initial conditions
obj_type = "cost" # "cost", "AZP"
x_wq_bounds = (0.2, 4)
u_wq_bounds = (0, 5)
x_wq_0 = 0.5 # reservoir water quality conditions
J = nothing # manually set number of discretization points
max_pump_switch = 5
# define different electricity tariff profiles
# c_elec = ones(network.n_t,1) # constant electricity tariff
c_elec = [ 99.0; 95.0; 91.0; 87.0; 89.0; 94.0; 109.0; 159.0; 229.0; 220.0; 180.0; 159.0; 156.0; 160.0; 175.0; 206.0; 245.0; 267.0; 253.0; 194.0; 162.0; 132.0; 110.0; 99.0 ]/1000 # Nord Pool UK prices on 13.12.2024 (GBP/kWh), see https://data.nordpoolgroup.com/auction/n2ex/prices?deliveryDate=2024-12-13&currency=GBP&aggregation=DeliveryPeriod&deliveryAreas=UK
opt_params = make_prob_data(network, Δt, Δk, sim_days, disc_method; pmin=pmin, QA=QA, x_wq_bounds=x_wq_bounds, u_wq_bounds=u_wq_bounds, obj_type=obj_type, x_wq_0=x_wq_0, J=J, max_pump_switch=max_pump_switch, c_elec=c_elec);

# c_r, c_j, c_tk, c_m, c_v, c_p, u = optimize_wq_fix_hyd(network, opt_results, sim_days, Δt, Δk, source_cl, b_loc, x0; kb=kb, kw=kw, disc_method=disc_method, x_bounds=x_bounds, u_bounds=u_bounds)

# run optimization solver
cpu_time = @elapsed begin
    solver = "Gurobi" # "Gurobi", "Ipopt"
    heuristic = false
    integer = true
    warm_start = false
    opt_results = optimize_hydraulic_wq(network, opt_params, sim_days, Δt, Δk, source_cl, b_loc, x_wq_0; x_wq_0=x_wq_0, solver=solver, integer=integer, warm_start=warm_start, heuristic=heuristic, optimize_wq=true, kb=kb, kw=kw, disc_method="implicit-upwind", x_bounds=x_wq_bounds, u_bounds=u_wq_bounds)
end

#= plot(opt_results.θ⁻[network.pump_idx[1], :])

plot(opt_results.q⁺[end, 1:end-1])
plot!(opt_results.q⁻[end, 1:end-1])

plot(opt_results.h_tk[1, 1:end-1])

plot(opt_results.h_j[1, 1:end-1]) =#