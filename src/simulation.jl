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





# """
# Water quality solver code from scratch.
# """
# function wq_solver()