##### Load dependencies and network data #####
using wq_robust_mpc
using Revise
using DataFrames
using LinearAlgebra
using SparseArrays
using LinearSolve
using LinearSolvePardiso
using Plots

net_name = "Net3" # "Threenode", "Net1", "Net3"
network = load_network(net_name)





##### Simulation functions #####

# EPANET solver
sim_type = "hydraulic" # "hydraulic", "chlorine", "age``, "trace"
sim_days = 7
if net_name == "Threenode" || net_name == "Net1"
    n_t = network.n_t * sim_days
elseif net_name == "Net3"
    n_t = network.n_t
end
sim_results = epanet_solver(network, sim_type; sim_days=sim_days)

# Water quality solver

# set hydraulic variables
# update link flow direction
A_i = repeat(hcat(network.A12, network.A10_res, network.A10_tank), 1, 1, n_t)
for k ∈ 1:n_t
    for link ∈ network.link_names
        link_idx = network.link_name_to_idx[link]
        if sim_results.flowdir[k, string(link)] == -1
            node_in = findall(x -> x == 1, A_i[link_idx, :, k])
            node_out = findall(x -> x == -1, A_i[link_idx, :, k])
            A_i[link_idx, node_in, k] .= -1
            A_i[link_idx, node_out, k] .= 1
        end
    end
end

# get flow and velocity values
q = Matrix(abs.(sim_results.flow[:, 2:end]) ./ 1000)' # convert to m^3/s
vel = Matrix(abs.(sim_results.velocity[:, 2:end]))'
d = Matrix(abs.(sim_results.demand[:, 2:end]) ./ 1000)' 

q_p = q[network.pipe_idx, :]
q_m = q[network.pump_idx, :]
q_v = q[network.valve_idx, :]
vel_p = vel[network.pipe_idx, :]

# get tank volumes
h_tk = sim_results.head[!, string.(network.node_names[network.tank_idx])]
lev_tk = h_tk .- repeat(network.elev[network.tank_idx], 1, n_t)'
V_tk = Matrix(lev_tk .* repeat(network.tank_area, 1, n_t)')'

# set discretization parameters and variables
Δt = 5 # 60 seconds
L_p = network.L[network.pipe_idx]
D_p = network.D[network.pipe_idx]
vel_p_max = maximum(vel_p, dims=2)
s_p = floor.(Int, L_p ./ (vel_p_max .* Δt))
s_p[s_p .== 0] .= 1
n_s = sum(s_p)
Δx_p = L_p ./ s_p
λ_p = vel_p ./ repeat(Δx_p, 1, n_t) .* Δt
λ_p[λ_p .> 1] .= 1 # bound λ to [0, 1]
λ_p[λ_p .< 0] .= 0 # bound λ to [0, 1]  

# initialize chlorine state variables
T = (sim_results.timestamp[end] + (sim_results.timestamp[end] - sim_results.timestamp[end-1])) .* 3600
T_k = Int(T ./ Δt) # number of discrete time steps
source_cl = repeat([1.5], network.n_r)

c_r_t = repeat(source_cl, 1)
c_j_t = zeros(network.n_j)
c_tk_t = zeros(network.n_tk)
c_m_t = zeros(network.n_m)
c_v_t = zeros(network.n_v)
c_p_t = zeros(n_s)
x_t = vcat(c_r_t, c_j_t, c_tk_t, c_m_t, c_v_t, c_p_t)

# set chlorine decay constants
kb = (0.5/3600/24) # units = 1/second
kw = (0.1/3600/24) # units = 1/second

### construct coefficient matrices across all hydraulic time steps ###
n_x = network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + n_s
n_u = 1 # control actuator inputs (TO BE COMPLETED)
k_t = 1 # iterate through hydraulic time steps later
k_t_Δt = 1 # iterate through hydraulic time steps later

u_t = zeros(n_u)

# Ex(t+Δt) = Ax(t) + Bu(t) + f(x(t))
E = spzeros(n_x, n_x)
A = spzeros(n_x, n_x)
B = spzeros(n_x, n_u)
f = spzeros(n_x, n_x)

# construct reservoir coefficient matrices
for r ∈ 1:network.n_r
    E[r, r] = 1.0
    A[r, r] = 1.0
end

# construct junction coefficient matrices
for (i, j) ∈ enumerate(network.junction_idx)

    # find all incoming and outgoing link indices at junction j
    I_in = findall(x -> x == 1, A_i[:, j, k_t_Δt]) # set of incoming links at junction j
    I_out = findall(x -> x == -1, A_i[:, j, k_t_Δt]) # set of outgoing links at junction j

    # assign c_j(t+Δt) matrix coefficients
    E[network.n_r + i, network.n_r + i] = 1.0
    # nothing for A matrix

    # assign c_link(t+Δt) matrix coefficients
    for link_idx ∈ I_in
        if link_idx ∈ network.pump_idx
            idx = findfirst(x -> x == link_idx, network.pump_idx)
            E[network.n_r + i, network.n_r + network.n_j + network.n_tk + idx] = -q_m[idx, k_t_Δt] ./ (d[j, k_t_Δt] + sum(q[I_out, k_t_Δt]))
        elseif link_idx ∈ network.valve_idx
            idx = findfirst(x -> x == link_idx, network.valve_idx)
            E[network.n_r + i, network.n_r + network.n_j + network.n_tk + network.n_m + idx] = -q_v[idx, k_t_Δt] ./ (d[j, k_t_Δt] + sum(q[I_out, k_t_Δt]))
        elseif link_idx ∈ network.pipe_idx
            idx = findfirst(x -> x == link_idx, network.pipe_idx)
            Δs = sum(s_p[1:idx])
            E[network.n_r + i, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs] = -q_p[idx, k_t_Δt] ./ (d[j, k_t_Δt] + sum(q[I_out, k_t_Δt]))
        else 
            @error "Link index not found in network pipe, pump, or valve indices."
        end
        # nothing for A matrix
    end
end


# construct tank coefficient matrices
for (i, tk) ∈ enumerate(network.tank_idx)

    # find all incoming and outgoing link indices at junction j
    I_in = findall(x -> x == 1, A_i[:, tk, k_t]) # set of incoming links at junction j
    I_out = findall(x -> x == -1, A_i[:, tk, k_t]) # set of outgoing links at junction j

    # assign c_tk(t+Δt) matrix coefficients
    E[network.n_r + network.n_j + i, network.n_r + network.n_j + i] = 1.0

    # assign c_tk(t) matrix coefficients
    A[network.n_r + network.n_j + i, network.n_r + network.n_j + i] = (V_tk[i, k_t] + - sum(q[I_out, k_t]) .* Δt) ./ V_tk[i, k_t_Δt]

    # assign c_link(t) matrix coefficients
    for link_idx ∈ I_in
        if link_idx ∈ network.pump_idx
            idx = findfirst(x -> x == link_idx, network.pump_idx)
            A[network.n_r + network.n_j + i, network.n_r + network.n_j + network.n_tk + idx] = q_m[idx, k_t] ./ V_tk[i, k_t_Δt]
        elseif link_idx ∈ network.valve_idx
            idx = findfirst(x -> x == link_idx, network.valve_idx)
            A[network.n_r + network.n_j + i, network.n_r + network.n_j + network.n_tk + network.n_m + idx] = q_v[idx, k_t] ./ V_tk[i, k_t_Δt]
        elseif link_idx ∈ network.pipe_idx
            idx = findfirst(x -> x == link_idx, network.pipe_idx)
            Δs = sum(s_p[1:idx])
            A[network.n_r + network.n_j + i, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs] = q_p[idx, k_t] ./ V_tk[i, k_t_Δt]
        else 
            @error "Link index not found in network pipe, pump, or valve indices."
        end
    end

    # assign decay coefficient in f for tank tk
    f[network.n_r + network.n_j + i, network.n_r + network.n_j + i] = kb * c_tk_t[i] * V_tk[i, k_t] * Δt * -1

end


# construct pump coefficient matrices
for (i, m) ∈ enumerate(network.pump_idx)

    # assign c_node(t+Δt) matrix coefficients
    node_out = findall(x -> x == -1, A_i[m, :, k_t_Δt])[1]
    if node_out ∈ network.reservoir_idx
        idx = findfirst(x -> x == node_out, network.reservoir_idx)
        E[network.n_r + network.n_j + network.n_tk + i, idx] = -1.0
    elseif node_out ∈ network.junction_idx
        idx = findfirst(x -> x == node_out, network.junction_idx)
        E[network.n_r + network.n_j + network.n_tk + i, network.n_r + idx] = -1.0
    elseif node_out ∈ network.tank_idx
        idx = findfirst(x -> x == node_out, network.tank_idx)
        E[network.n_r + network.n_j + network.n_tk + i, network.n_r + network.n_j + idx] = -1.0
    else
        @error "Pump upstream node not found in network reservoir, junction, or tank indices."
    end

    # assign c_pump(t+Δt) matrix coefficients
    E[network.n_r + network.n_j + network.n_tk + i, network.n_r + network.n_j + network.n_tk + i] = 1.0

end


# construct valve coefficient matrices
for (i, v) ∈ enumerate(network.valve_idx)

    # assign c_node(t+Δt) matrix coefficients
    node_out = findall(x -> x == -1, A_i[v, :, k_t_Δt])[1]
    if node_out ∈ network.reservoir_idx
        idx = findfirst(x -> x == node_out, network.reservoir_idx)
        E[network.n_r + network.n_j + network.n_tk + network.n_m + i, idx] = -1.0
    elseif node_out ∈ network.junction_idx
        idx = findfirst(x -> x == node_out, network.junction_idx)
        E[network.n_r + network.n_j + network.n_tk + network.n_m + i, network.n_r + idx] = -1.0
    elseif node_out ∈ network.tank_idx
        idx = findfirst(x -> x == node_out, network.tank_idx)
        E[network.n_r + network.n_j + network.n_tk + network.n_m + i, network.n_r + network.n_j + idx] = -1.0
    else
        @error "Valve upstream node not found in network reservoir, junction, or tank indices."
    end

    # assign c_valve(t+Δt) matrix coefficients
    E[network.n_r + network.n_j + network.n_tk + network.n_m + i, network.n_r + network.n_j + network.n_tk + network.n_m + i] = 1.0

end


# construct pipe coefficient matrices (start with Lax-Wendroff discretization scheme)
λ_p1 = 0.5 .* λ_p[:, k_t] .* (1 .+ λ_p[:, k_t])
λ_p2 = (1 .- λ_p[:, k_t]).^2
λ_p3 = -0.5 .* λ_p[:, k_t] .* (1 .- λ_p[:, k_t])

# mass transfer coefficient data
ν = 1.0533e-6 # in m^2/s
D_m = 1.208e-9 # @ 20°C in m^2/s
Sc = ν / D_m
Re = (4 .* q_p[:, k_t]) ./ (π .* D_p .* ν)

for (i, p) ∈ enumerate(network.pipe_idx)

    # compute mass transfer coefficient for pipe p (based on EPANET manual first-order decay model)
    if Re[i] < 2000
        Sh = 3.65 + (0.0668 * (D_p[i]/L_p[i]) * Re[i] * Sc) /  (1 + 0.04 * ((D_p[i]/L_p[i]) * Re[i] * Sc)^(2/3))
    else
        Sh = 0.0149 * Re[i]^0.88 * Sc^(1/3)
    end
    kf = Sh *(D_m / D_p[i])

    # find upstream and downstream node indices at pipe p
    node_in = findall(x -> x == 1, A_i[p, :, kt])[1]
    node_out = findall(x -> x == -1, A_i[p, :, kt])[1]

    for s ∈ 1:s_p[i]

        Δs = sum(s_p[1:i-1])

        # assign C_pipe(t+Δt) matrix coefficients
        E[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = 1.0

        # assign decay term in f for pipe p
        k_p = (kb + (4 * kw * kf * c_p_t[Δs + s]) / (D_p[i] * (kw + kf)))
        f[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = k_p * c_p_t[Δs + s] * Δt * -1

        # assign C_pipe(t) and C_juction(t) matrix coefficients
        if s == 1

            # previous segment s-1 matrix coefficients at time t
            if node_out ∈ network.reservoir_idx
                idx = findfirst(x -> x == node_out, network.reservoir_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, idx] = λ_p1[i]
            elseif node_out ∈ network.junction_idx
                idx = findfirst(x -> x == node_out, network.junction_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + idx] = λ_p1[i]
            elseif node_out ∈ network.tank_idx
                idx = findfirst(x -> x == node_out, network.tank_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + idx] = λ_p1[i]
            else
                @error "Upstream node of pipe p not found in network reservoir, junction, or tank indices."
            end

            # current segment s matrix coefficients at time t
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = λ_p2[i]

            # next segment s+1 matrix coefficients at time t
            if s_p[i] == 1
                if node_in ∈ network.reservoir_idx
                    idx = findfirst(x -> x == node_in, network.reservoir_idx)
                    A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, idx] = λ_p3[i]
                elseif node_in ∈ network.junction_idx
                    idx = findfirst(x -> x == node_in, network.junction_idx)
                    A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + idx] = λ_p3[i]
                elseif node_in ∈ network.tank_idx
                    idx = findfirst(x -> x == node_in, network.tank_idx)
                    A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + idx] = λ_p3[i]
                else
                    @error "Downstream node of pipe p not found in network reservoir, junction, or tank indices."
                end
            else
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s + 1] = λ_p3[i]
            end

        elseif s == s_p[i] && s > 1

            # previous segment s-1 matrix coefficients at time t
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s - 1] = λ_p1[i]

            # current segment s matrix coefficients at time t
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = λ_p2[i]

            # next segment s+1 matrix coefficients at time t
            if node_in ∈ network.reservoir_idx
                idx = findfirst(x -> x == node_in, network.reservoir_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, idx] = λ_p3[i]
            elseif node_in ∈ network.junction_idx
                idx = findfirst(x -> x == node_in, network.junction_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + idx] = λ_p3[i]
            elseif node_in ∈ network.tank_idx
                idx = findfirst(x -> x == node_in, network.tank_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + idx] = λ_p3[i]
            else
                @error "Downstream node of pipe p not found in network reservoir, junction, or tank indices."
            end

        else

            # previous segment s-1 matrix coefficients at time t
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s - 1] = λ_p1[i]

            # current segment s matrix coefficients at time t
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = λ_p2[i]

            # next segment s+1 matrix coefficients at time t
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s + 1] = λ_p3[i]
            
        end
    end

end



# form system of linear equations: Ex(t+Δt) = Ax(t) + Bu(t) + f(x(t))
A_x_t = A * x_t
B_u_t = B * u_t
f_x_t = f * x_t
cpu_time = @elapsed begin
    # prob = LinearProblem(E, A_x_t + B_u_t + f_x_t)
    # xt_Δt = solve(prob, MKLPardisoFactorize()).u
    # linsolve = init(prob)
    # xt_Δt = solve(linsolve)
    x_t_Δt = E \ (A_x_t + B_u_t + f_x_t)
end


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
