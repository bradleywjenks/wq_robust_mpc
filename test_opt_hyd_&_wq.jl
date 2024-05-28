##### Load dependencies and network data, and optimization parameters #####
using wq_robust_mpc
using Revise
using Plots


# load network data
net_name = "Net1" # "Threenode", "Net1", "Net3", "2loopsNet", "Net25"
network = load_network(net_name)

# create optimization parameters
sim_days = 1
Δk = 60 * 60
Δt = 60 * 60
QA = true
pmin = 15
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
obj_type = "AZP"
x_wq_bounds = (0.2, 4)
u_wq_bounds = (0, 5)
opt_params = make_prob_data(network, Δt, Δk, sim_days, disc_method; pmin=pmin, QA=QA, x_wq_bounds=x_wq_bounds, u_wq_bounds=u_wq_bounds)

x_wq_0 = 0.5 # initial water quality conditions
opt_results = optimize_hydraulic_wq(network, opt_params; x_wq_0=x_wq_0)

