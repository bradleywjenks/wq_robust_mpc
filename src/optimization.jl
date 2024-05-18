"""
Collection of functions for optimizing water quality
"""

using PyCall
using DataFrames
using LinearAlgebra
using SparseArrays
using JuMP
using Ipopt
using Gurobi





"""
Main function for optimizing water quality
"""
function optimize_wq(network, sim_days, Δt, Δk, source_cl, b_loc, x0; kb=0.5, kw=0.1, disc_method="explicit-central", x_bounds=(0.5, 3), u_bounds=(0, 5))


    ##### SET OPTIMIZATION PARAMETERS #####

    # assign constant parameters
    kb = (kb/3600/24) # units = 1/second
    kw = (kw/3600/24) # units = 1/second
    ν = 1.0533e-6 # kinematic velocity of water in m^2/s
    D_m = 1.208e-9 # molecular diffusivity of chlorine @ 20°C in m^2/s
    Sc = ν / D_m # schmidt number for computing mass transfer coefficient
    ϵ_reg = 1e-3 # small regularization value to avoid division by zero

    # unload network data
    net_name = network.name
    n_r = network.n_r
    n_j = network.n_j
    n_tk = network.n_tk
    n_m = network.n_m
    n_v = network.n_v
    n_p = network.n_p
    reservoir_idx = network.reservoir_idx
    junction_idx = network.junction_idx
    tank_idx = network.tank_idx
    pump_idx = network.pump_idx
    valve_idx = network.valve_idx
    pipe_idx = network.pipe_idx
    link_names = network.link_names
    node_names = network.node_names
    link_name_to_idx = network.link_name_to_idx
    node_name_to_idx = network.node_name_to_idx
    L_p = network.L[pipe_idx]
    D_p = network.D[pipe_idx]

    # compute network hydraulics
    sim_type = "hydraulic"
    sim_results = epanet_solver(network, sim_type; sim_days=sim_days, Δk=Δk)
    k_set = sim_results.timestamp .* 3600
    n_t = size(k_set)[1]

    # get flow, velocity, demand, and Reynolds number values
    q = Matrix((sim_results.flow[:, 2:end]))'
    qdir = zeros(size(q, 1), size(q, 2))
    qdir[q .> 1e-3] .= 1
    qdir[q .< -1e-3] .= -1
    for row in eachrow(qdir)
        for t in 1:n_t
            # march forward in time
            if t != 1 && row[t] == 0
                row[t] = row[t-1]
            end
            # march backward in time
            if t != n_t && row[end - t] == 0
                row[end-t] = row[end+1-t]
            end
        end
    end
    q = abs.(q)
    vel = Matrix(abs.(sim_results.velocity[:, 2:end]))'
    d = Matrix(abs.(sim_results.demand[:, 2:end]))' 

    q_p = q[pipe_idx, :]
    q_m = q[pump_idx, :]
    q_v = q[valve_idx, :]
    vel_p = vel[pipe_idx, :]
    Re = (4 .* (q_p ./ 1000)) ./ (π .* D_p .* ν)

    # update link flow direction
    A_inc_0 = repeat(hcat(network.A12, network.A10_res, network.A10_tank), 1, 1, n_t)
    A_inc = copy(A_inc_0)
    for k ∈ 1:n_t
        for link ∈ link_names
            link_idx = link_name_to_idx[link]
            if qdir[link_idx, k] == -1
                node_in = findall(x -> x == 1, A_inc[link_idx, :, k])
                node_out = findall(x -> x == -1, A_inc[link_idx, :, k])
                A_inc[link_idx, node_in, k] .= -1
                A_inc[link_idx, node_out, k] .= 1
            end
        end
    end

    # get tank volumes
    h_tk = sim_results.head[!, string.(node_names[tank_idx])]
    lev_tk = h_tk .- repeat(network.elev[tank_idx], 1, n_t)'
    V_tk = Matrix(lev_tk .* repeat(network.tank_area, 1, n_t)')' .* 1000 # convert to L

    # set discretization parameters and variables
    vel_p_max = maximum(vel_p, dims=2)
    s_p = L_p ./ (vel_p_max .* Δt)
    if any(s_p .< 0.75)
        @error "At least one pipe has discretization step Δx > pipe length. Please input a smaller Δt."
        return
    end
    s_p = floor.(Int, s_p)
    s_p[s_p .== 0] .= 1
    n_s = sum(s_p)
    Δx_p = L_p ./ s_p
    λ_p = vel_p ./ repeat(Δx_p, 1, n_t) .* Δt

    # check CFL condition
    for k ∈ 1:n_t
        if any(Δt .>= Δx_p ./ vel_p[:, k])
            @error "CFL condition not satisfied. Please input a smaller Δt."
            return s_p
        end
    end

    # simulation times
    T = round(sim_results.timestamp[end] + (sim_results.timestamp[end] - sim_results.timestamp[end-1]), digits=0) * 3600
    T_k = Int(T / Δt) # number of discrete time steps


    
    ##### CREATE OPTIMIZATION MODEL #####

    model = Model(Gurobi.Optimizer)
    # set_silent(model)

    # define variables
    @variable(model, x_bounds[1] ≤ c_r[i=1:n_r, t=1:T_k] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_j[i=1:n_j, t=1:T_k] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_tk[i=1:n_tk, t=1:T_k] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_m[i=1:n_m, t=1:T_k] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_v[i=1:n_v, t=1:T_k] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_p[i=1:n_s, t=1:T_k] ≤ x_bounds[2])

    # define constraints
    # insert constraints here... 


    x = nothing
    u = nothing

    return x, u


end

