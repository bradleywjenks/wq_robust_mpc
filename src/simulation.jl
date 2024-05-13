"""
Collection of functions for simulating hydraulic and water quality states
"""

using PyCall
using DataFrames
using InlineStrings
using AutomaticDocstrings
using Dates
using Parameters
using Random
using Distributions
using SparseArrays
using LinearAlgebra
using LinearSolve
using LinearSolvePardiso


@with_kw mutable struct SimResults
    timestamp::Union{Vector{Float64}, Nothing} = nothing
    pressure::Union{DataFrame, Nothing} = nothing
    head::Union{DataFrame, Nothing} = nothing
    demand::Union{DataFrame, Nothing} = nothing
    flow::Union{DataFrame, Nothing} = nothing
    flowdir::Union{DataFrame, Nothing} = nothing
    velocity::Union{DataFrame, Nothing} = nothing
    chlorine::Union{DataFrame, Nothing} = nothing
    age::Union{DataFrame, Nothing} = nothing
    trace::Union{DataFrame, Nothing} = nothing
end

Base.copy(r::SimResults) = SimResults(
    timestamp=deepcopy(r.timestamp),
    pressure=deepcopy(r.pressure),
    head=deepcopy(r.head),
    demand=deepcopy(r.demand),
    flow=deepcopy(r.flow),
    flowdir=deepcopy(r.flowdir),
    velocity=deepcopy(r.velocity),
    chlorine=deepcopy(r.chlorine),
    age=deepcopy(r.age),
    trace=deepcopy(r.trace),
)


"""
Function for getting the number of hydraulic time steps when simulation period is modified.
"""
function get_hydraulic_time_steps(network, net_name, sim_days)
    if net_name == "Threenode" || net_name == "Net1"
        n_t = network.n_t * sim_days
    elseif net_name == "Net3"
        n_t = network.n_t
    else 
        n_t = nothing
        @error "Network name not recognized."
    end
    return n_t
end


"""
Main function for calling EPANET solver via WNTR Python package
"""
function epanet_solver(network::Network, sim_type; prv_settings=nothing, afv_settings=nothing, cl_0=repeat([1.5], size(network.reservoir_idx, 1)), trace_node=network.node_names[network.reservoir_idx[1]], sim_days=7)

    net_path = NETWORK_PATH * network.name * "/" * network.name * ".inp"

    # load network file via WNTR pacakge
    wntr = pyimport("wntr")
    wn = wntr.network.WaterNetworkModel(net_path)

    # set simulation times
    wn.options.time.duration = sim_days * 24 * 3600 # number of days set by user
    wn.options.time.hydraulic_timestep = 3600 # 1-hour hydraulic time step
    wn.options.time.quality_timestep = 60 * 5 # 5-minute water quality time step

    # # function for setting PRV settings
    # wn = set_pcv_settings(wntr, wn, network, prv_settings)

    # # function for setting AFV settings
    # wn = set_afv_settings(wn, wntr, network, afv_settings)

    # wn.convert_controls_to_rules(priority=3)

    # run hydraulic or water quality simulation
    if sim_type == "hydraulic"
        sim_results = epanet_hydraulic(network, wntr, wn)
    elseif sim_type ∈ ["chlorine", "age", "trace"]
        sim_results = epanet_wq(network, wntr, wn, sim_type, trace_node, cl_0)
    else
        @error "Simulation type not recognized."
        sim_results = SimResults()
    end

    return sim_results
    
end


"""
Hydraulic simulation code.
"""
function epanet_hydraulic(network, wntr, wn)

    node_names = network.node_names

    # set time column
    time_step = wn.options.time.report_timestep
    duration = wn.options.time.duration
    sim_time = collect(0:time_step/3600:duration/3600)

    # run hydraulic solver
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # obtain flow at links
    df_flow = DataFrame()
    df_flow.timestamp = sim_time
    for col ∈ results.link["flowrate"].columns
        df_flow[!, col] = abs.(getproperty(results.link["flowrate"], col).values .* 1000) # convert to Lps
    end

    # obtain link flow direction
    df_flowdir = DataFrame()
    df_flowdir.timestamp = sim_time
    for col ∈ results.link["flowrate"].columns
        df_flowdir[!, col] = sign.(getproperty(results.link["flowrate"], col).values)
    end

    # obtain velocity at links
    df_vel = DataFrame()
    df_vel.timestamp = sim_time
    for col ∈ results.link["velocity"].columns
        df_vel[!, col] = getproperty(results.link["velocity"], col).values
    end

    # obtain pressure at nodes
    df_pressure = DataFrame()
    df_pressure.timestamp = sim_time
    for col ∈ results.node["pressure"].columns
        if col ∈ string.(node_names[vcat(network.reservoir_idx, network.tank_idx)])
            df_pressure[!, col] = zeros(size(sim_time))
        else
            df_pressure[!, col] = getproperty(results.node["pressure"], col).values
        end
    end
    # atmospheric_nodes = [col for col in names(df_pressure) if col in node_names[vcat(network.reservoir_idx, network.tank_idx)]]
    # println(atmospheric_nodes)
    # df_pressure[!, atmospheric_nodes] .= 0 # set reservoir and tank pressures to 0

    # obtain head at nodes
    df_head = DataFrame()
    df_head.timestamp = sim_time
    for col ∈ results.node["head"].columns
        df_head[!, col] = getproperty(results.node["head"], col).values
    end

    # obtain demand at nodes
    df_demand = DataFrame()
    df_demand.timestamp = sim_time
    for col ∈ results.node["demand"].columns
        df_demand[!, col] = getproperty(results.node["demand"], col).values .* 1000 # convert to Lps
    end


    # create simulation results structure
    sim_results = SimResults(
        timestamp=sim_time[1:end-1],
        pressure=df_pressure[1:end-1, :],
        head=df_head[1:end-1, :],
        demand=df_demand[1:end-1, :],
        flow=df_flow[1:end-1, :],
        flowdir=df_flowdir[1:end-1, :],
        velocity=df_vel[1:end-1, :],
    )

    return sim_results

end




"""
Water quality simulation code via EPANET solver.
"""
function epanet_wq(network, wntr, wn, sim_type, trace_node, cl_0)

    # set time column
    time_step = wn.options.time.report_timestep
    duration = wn.options.time.duration
    sim_time = collect(0:time_step/3600:duration/3600)

    # select water quality solver
    if sim_type == "chlorine"

        wn.options.quality.parameter = "CHEMICAL"

        # set source chlorine values
        for (idx, name) in enumerate(network.node_names[network.reservoir_idx])
            # wn.add_pattern(name * "_cl", res_vals[:, idx])
            wn.add_source(string(name) * "_cl", string(name), "CONCEN", cl_0[idx])
        end

        # set chlorine reaction parameters
        wn.options.reaction.bulk_coeff = (-0.5/3600/24) # units = 1/second
        wn.options.reaction.wall_coeff = (-0.1/3600/24) # units = 1/second

    elseif sim_type == "age"

        wn.options.quality.parameter = "AGE"

    elseif sim_type == "trace"

        wn.options.quality.parameter = "TRACE"
        wn.options.quality.trace_node = trace_node

    end

    # run water quality solver
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # obtain water quality (chlorine or age) at nodes
    df_qual = DataFrame()
    df_qual.timestamp = sim_time
    if sim_type == "chlorine"
        for col ∈ results.node["quality"].columns
            df_qual[!, col] = getproperty(results.node["quality"], col).values
        end

        sim_results = SimResults(
            timestamp=sim_time[1:end-1],
            chlorine=df_qual[1:end-1, :]
        )
    elseif sim_type == "age"
        for col ∈ results.node["quality"].columns
            df_qual[!, col] = getproperty(results.node["quality"], col).values ./3600
        end

        sim_results = SimResults(
            timestamp=sim_time[1:end-1],
            age=df_qual[1:end-1, :]
        )

    elseif sim_type == "trace"
        for col ∈ results.node["quality"].columns
            df_qual[!, col] = getproperty(results.node["quality"], col).values
        end

        sim_results = SimResults(
            timestamp=sim_time[1:end-1],
            trace=df_qual[1:end-1, :]
        )
    end

    return sim_results

end





"""
Water quality solver code from scratch.
"""
function wq_solver(network, sim_days, Δt, source_cl, u; kb=0.5, kw=0.1, disc_method="explicit-central")

    ##### INITIALIZE WQ_SOLVER PARAMETERS #####

    # assign constant parameters
    # kb = (kb/3600/24) # units = 1/second
    kb = 0
    kw = (kw/3600/24) # units = 1/second
    ν = 1.0533e-6 # kinematic velocity of water in m^2/s
    D_m = 1.208e-9 # molecular diffusivity of chlorine @ 20°C in m^2/s
    Sc = ν / D_m # schmidt number for computing mass transfer coefficient

    # unload network data
    net_name = network.name
    n_t = get_hydraulic_time_steps(network, net_name, sim_days)
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
    sim_results = epanet_solver(network, sim_type; sim_days=sim_days)
    k_set = sim_results.timestamp .* 3600

    # update link flow direction
    A_inc = repeat(hcat(network.A12, network.A10_res, network.A10_tank), 1, 1, n_t)
    for k ∈ 1:n_t
        for link ∈ link_names
            link_idx = link_name_to_idx[link]
            if sim_results.flowdir[k, string(link)] == -1
                node_in = findall(x -> x == 1, A_inc[link_idx, :, k])
                node_out = findall(x -> x == -1, A_inc[link_idx, :, k])
                A_inc[link_idx, node_in, k] .= -1
                A_inc[link_idx, node_out, k] .= 1
            end
        end
    end

    # get flow, velocity, demand, and Reynolds number values
    q = Matrix(abs.(sim_results.flow[:, 2:end]) ./ 1000)' # convert to m^3/s
    vel = Matrix(abs.(sim_results.velocity[:, 2:end]))'
    d = Matrix(abs.(sim_results.demand[:, 2:end]) ./ 1000)' 

    q_p = q[pipe_idx, :]
    q_m = q[pump_idx, :]
    q_v = q[valve_idx, :]
    vel_p = vel[pipe_idx, :]
    Re = (4 .* q_p) ./ (π .* D_p .* ν)

    # get tank volumes
    h_tk = sim_results.head[!, string.(node_names[tank_idx])]
    lev_tk = h_tk .- repeat(network.elev[tank_idx], 1, n_t)'
    V_tk = Matrix(lev_tk .* repeat(network.tank_area, 1, n_t)')'

    # set discretization parameters and variables
    vel_p_max = maximum(vel_p, dims=2)
    s_p = floor.(Int, L_p ./ (vel_p_max .* Δt))
    if any(s_p .< 1)
        @error "Water quality time step Δt is too large. Please decrease Δt."
    end
    # s_p[s_p .== 0] .= 1
    n_s = sum(s_p)
    Δx_p = L_p ./ s_p
    λ_p = vel_p ./ repeat(Δx_p, 1, n_t) .* Δt
    λ_p[λ_p .> 1] .= 1 # bound λ to [0, 1]
    λ_p[λ_p .< 0] .= 0 # bound λ to [0, 1]  

    # initialize chlorine state variables
    T = (sim_results.timestamp[end] + (sim_results.timestamp[end] - sim_results.timestamp[end-1])) .* 3600
    T_k = Int(T / Δt) # number of discrete time steps
    c_r_t = source_cl
    c_j_t = zeros(n_j)
    c_tk_t = zeros(n_tk)
    c_m_t = zeros(n_m)
    c_v_t = zeros(n_v)
    c_p_t = zeros(n_s)
    x_t = vcat(c_r_t, c_j_t, c_tk_t, c_m_t, c_v_t, c_p_t)

    n_x = n_r + n_j + n_tk + n_m + n_v + n_s
    n_u = size(u, 1) # control actuator inputs

    # initialize coefficient matrices in system of linear equations
    # Ex(t+Δt) = Ax(t) + Bu(t) + f(x(t))
    E = spzeros(n_x, n_x)
    A = spzeros(n_x, n_x)
    B = spzeros(n_x, n_u)
    f = spzeros(n_x, n_x)

    # initialize solution matrix
    x = zeros(n_x, T_k+1)
    x[:, 1] .= x_t

    

    ##### MAIN WQ_SOLVER LOOP #####
    for t ∈ 1:T_k

        @info "Solving water quality states at time step t = $t"

        # find hydraulic time step index
        k_t = searchsortedfirst(k_set, t) - 1
        k_t_Δt = searchsortedfirst(k_set, t+1) - 1

        # construct reservoir coefficient matrices
        for r ∈ 1:n_r
            E[r, r] = 1
            A[r, r] = 1
        end

        # construct junction coefficient matrices
        for (i, j) ∈ enumerate(junction_idx)

            # find all incoming and outgoing link indices at junction j
            I_in = findall(x -> x == 1, A_inc[:, j, k_t_Δt]) # set of incoming links at junction j
            I_out = findall(x -> x == -1, A_inc[:, j, k_t_Δt]) # set of outgoing links at junction j

            # assign c_j(t+Δt) matrix coefficients
            E[n_r + i, n_r + i] = 1
            # nothing for A matrix

            # assign c_link(t+Δt) matrix coefficients
            for link_idx ∈ I_in
                if link_idx ∈ pump_idx
                    idx = findfirst(x -> x == link_idx, pump_idx)
                    E[n_r + i, n_r + n_j + n_tk + idx] = -q_m[idx, k_t_Δt] ./ (d[j, k_t_Δt] + sum(q[I_out, k_t_Δt]))
                elseif link_idx ∈ valve_idx
                    idx = findfirst(x -> x == link_idx, valve_idx)
                    E[n_r + i, n_r + n_j + n_tk + n_m + idx] = -q_v[idx, k_t_Δt] ./ (d[j, k_t_Δt] + sum(q[I_out, k_t_Δt]))
                elseif link_idx ∈ pipe_idx
                    idx = findfirst(x -> x == link_idx, pipe_idx)
                    Δs = sum(s_p[1:idx])
                    E[n_r + i, n_r + n_j + n_tk + n_m + n_v + Δs] = -q_p[idx, k_t_Δt] ./ (d[j, k_t_Δt] + sum(q[I_out, k_t_Δt]))
                else 
                    @error "Link index $link_index not found in network pipe, pump, or valve indices."
                end
                # nothing for A matrix
            end
        end

        # construct tank coefficient matrices
        for (i, tk) ∈ enumerate(tank_idx)

            # find all incoming and outgoing link indices at junction j
            I_in = findall(x -> x == 1, A_inc[:, tk, k_t]) # set of incoming links at junction j
            I_out = findall(x -> x == -1, A_inc[:, tk, k_t]) # set of outgoing links at junction j

            # assign c_tk(t+Δt) matrix coefficients
            E[n_r + n_j + i, n_r + n_j + i] = 1

            # assign c_tk(t) matrix coefficients
            A[n_r + n_j + i, n_r + n_j + i] = (V_tk[i, k_t] - sum(q[I_out, k_t]) .* Δt) ./ V_tk[i, k_t_Δt]

            # assign c_link(t) matrix coefficients
            for link_idx ∈ I_in
                if link_idx ∈ pump_idx
                    idx = findfirst(x -> x == link_idx, pump_idx)
                    A[n_r + n_j + i, n_r + n_j + n_tk + idx] = q_m[idx, k_t] ./ V_tk[i, k_t_Δt]
                elseif link_idx ∈ valve_idx
                    idx = findfirst(x -> x == link_idx, valve_idx)
                    A[n_r + n_j + i, n_r + n_j + n_tk + n_m + idx] = q_v[idx, k_t] ./ V_tk[i, k_t_Δt]
                elseif link_idx ∈ pipe_idx
                    idx = findfirst(x -> x == link_idx, pipe_idx)
                    Δs = sum(s_p[1:idx])
                    A[n_r + n_j + i, n_r + n_j + n_tk + n_m + n_v + Δs] = q_p[idx, k_t] ./ V_tk[i, k_t_Δt]
                else 
                    @error "Link index $link_index not found in network pipe, pump, or valve indices."
                end
            end

            # assign decay coefficient in f for tank tk
            f[n_r + n_j + i, n_r + n_j + i] = kb * V_tk[i, k_t] * Δt * -1

        end

        # construct pump coefficient matrices
        for (i, m) ∈ enumerate(pump_idx)

            # assign c_node(t+Δt) matrix coefficients
            node_out = findall(x -> x == -1, A_inc[m, :, k_t_Δt])[1]
            if node_out ∈ reservoir_idx
                idx = findfirst(x -> x == node_out, reservoir_idx)
                E[n_r + n_j + n_tk + i, idx] = -1
            elseif node_out ∈ junction_idx
                idx = findfirst(x -> x == node_out, junction_idx)
                E[n_r + n_j + n_tk + i, n_r + idx] = -1
            elseif node_out ∈ tank_idx
                idx = findfirst(x -> x == node_out, tank_idx)
                E[n_r + n_j + n_tk + i, n_r + n_j + idx] = -1
            else
                @error "Pump upstream node $node_out not found in network reservoir, junction, or tank indices."
            end

            # assign c_pump(t+Δt) matrix coefficients
            E[n_r + n_j + n_tk + i, n_r + n_j + n_tk + i] = 1

        end

        # construct valve coefficient matrices
        for (i, v) ∈ enumerate(valve_idx)

            # assign c_node(t+Δt) matrix coefficients
            node_out = findall(x -> x == -1, A_inc[v, :, k_t_Δt])[1]
            if node_out ∈ reservoir_idx
                idx = findfirst(x -> x == node_out, reservoir_idx)
                E[n_r + n_j + n_tk + n_m + i, idx] = -1.0
            elseif node_out ∈ junction_idx
                idx = findfirst(x -> x == node_out, junction_idx)
                E[n_r + n_j + n_tk + n_m + i, n_r + idx] = -1.0
            elseif node_out ∈ tank_idx
                idx = findfirst(x -> x == node_out, tank_idx)
                E[n_r + n_j + n_tk + n_m + i, n_r + n_j + idx] = -1.0
            else
                @error "Valve upstream node $node_out not found in network reservoir, junction, or tank indices."
            end

            # assign c_valve(t+Δt) matrix coefficients
            E[n_r + n_j + n_tk + n_m + i, n_r + n_j + n_tk + n_m + i] = 1.0

        end

        # construct pipe (segment) coefficient matrices
        for (i, p) ∈ enumerate(pipe_idx)

            # compute mass transfer coefficient for pipe p (based on EPANET manual first-order decay model)
            if Re[i, k_t] < 2000
                Sh = 3.65 + (0.0668 * (D_p[i]/L_p[i]) * Re[i, k_t] * Sc) /  (1 + 0.04 * ((D_p[i]/L_p[i]) * Re[i, k_t] * Sc)^(2/3))
            else
                Sh = 0.0149 * Re[i, k_t]^0.88 * Sc^(1/3)
            end
            kf = Sh *(D_m / D_p[i])
        
            # find upstream and downstream node indices at pipe p
            node_in = findall(x -> x == 1, A_inc[p, :, k_t])[1]
            node_out = findall(x -> x == -1, A_inc[p, :, k_t])[1]
        
            for s ∈ 1:s_p[i]
        
                Δs = sum(s_p[1:i-1])
        
                # assign decay term in f for pipe p
                # k_p = kb + ((4 * kw * kf * x_t[n_r + n_j + n_tk + n_m + n_v + Δs + s]) / (D_p[i] * (kw + kf)))
                k_p = kb
                f[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = k_p * Δt * -1

                # assign C_pipe(t+Δt) and C_pipe(t) matrix coefficients
                if disc_method == "explicit-central"
                    E, A = explicit_central_disc(network, E, A, Δs, s, i, s_p, λ_p, k_t, node_in, node_out)
                elseif disc_method == "implicit-central"
                    @error "Implicit-central discretization method not yet implemented."
                    # E, A = implicit_central_disc(network, E, A, s, i, n_s, s_p, λ_p, k_t)
                else
                    @error "Discretization method not recognized."
                end

            end
        
        end

        # solve system of linear equations
        A_x_t = A * x_t
        B_u_t = B * u[:, k_t]
        f_x_t = f * x_t
        x_t_Δt = E \ (A_x_t + B_u_t + f_x_t)
        x[:, t+1] .= x_t_Δt
        # prob = LinearProblem(E, A_x_t + B_u_t + f_x_t)
        # xt_Δt = solve(prob, MKLPardisoFactorize()).u
        # linsolve = init(prob)
 
        x_t = x_t_Δt
    end

    return x

end




"""
Explicit-central (Lax-Wendroff) discretization method for pipe segment s @ time step t and t+Δt
"""
function explicit_central_disc(network, E, A, Δs, s, i, s_p, λ_p, k_t, node_in, node_out)

    # get discretization parameters for pipe i at hydraulic time step k
    λ_p1 = 0.5 * λ_p[i, k_t] * (1 + λ_p[i, k_t])
    λ_p2 = (1 - λ_p[i, k_t])^2
    λ_p3 = -0.5 * λ_p[i, k_t] * (1 - λ_p[i, k_t])

    E[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = 1
        
    # assign C_pipe(t) and C_juction(t) matrix coefficients
    if s == 1

        # previous segment s-1 matrix coefficients at time t
        if node_out ∈ network.reservoir_idx
            idx = findfirst(x -> x == node_out, network.reservoir_idx)
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, idx] = λ_p1
        elseif node_out ∈ network.junction_idx
            idx = findfirst(x -> x == node_out, network.junction_idx)
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + idx] = λ_p1
        elseif node_out ∈ network.tank_idx
            idx = findfirst(x -> x == node_out, network.tank_idx)
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + idx] = λ_p1
        else
            @error "Upstream node of pipe $i not found in network reservoir, junction, or tank indices."
        end

        # current segment s matrix coefficients at time t
        A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = λ_p2

        # next segment s+1 matrix coefficients at time t
        if s_p[i] == 1
            if node_in ∈ network.reservoir_idx
                idx = findfirst(x -> x == node_in, network.reservoir_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, idx] = λ_p3
            elseif node_in ∈ network.junction_idx
                idx = findfirst(x -> x == node_in, network.junction_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + idx] = λ_p3
            elseif node_in ∈ network.tank_idx
                idx = findfirst(x -> x == node_in, network.tank_idx)
                A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + idx] = λ_p3
            else
                @error "Downstream node of pipe $i not found in network reservoir, junction, or tank indices."
            end
        else
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s + 1] = λ_p3
        end

    elseif s == s_p[i] && s > 1

        # previous segment s-1 matrix coefficients at time t
        A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s - 1] = λ_p1

        # current segment s matrix coefficients at time t
        A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = λ_p2

        # next segment s+1 matrix coefficients at time t
        if node_in ∈ network.reservoir_idx
            idx = findfirst(x -> x == node_in, network.reservoir_idx)
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, idx] = λ_p3
        elseif node_in ∈ network.junction_idx
            idx = findfirst(x -> x == node_in, network.junction_idx)
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + idx] = λ_p3
        elseif node_in ∈ network.tank_idx
            idx = findfirst(x -> x == node_in, network.tank_idx)
            A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + idx] = λ_p3
        else
            @error "Downstream node of pipe $i not found in network reservoir, junction, or tank indices."
        end

    else

        # previous segment s-1 matrix coefficients at time t
        A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s - 1] = λ_p1

        # current segment s matrix coefficients at time t
        A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s] = λ_p2

        # next segment s+1 matrix coefficients at time t
        A[network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s, network.n_r + network.n_j + network.n_tk + network.n_m + network.n_v + Δs + s + 1] = λ_p3
        
    end


    return E, A

end