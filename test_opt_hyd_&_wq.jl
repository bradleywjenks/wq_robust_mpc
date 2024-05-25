##### Load dependencies and network data, and optimization parameters #####
using wq_robust_mpc
using Revise
using Plots


# load network data
net_name = "Net1" # "Threenode", "Net1", "Net3"
network = load_network(net_name)

# create optimization parameters
sim_days = 7
Δk = 60 * 60
Δt = 60 * 60
QA = true
pmin = 15
disc_method = "implicit-upwind" # "explicit-central", "implicit-upwind"
opt_data = make_prob_data(network, Δt, Δk, sim_days, disc_method; pmin=pmin, QA=QA)

x_hyd, x_wq, u_pump, u_booster = optimize_hydraulic_wq(network, opt_data)