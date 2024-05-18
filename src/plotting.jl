"""
Collection of plotting functions.
"""

using GraphPlot
using Graphs
using GraphMakie
using CairoMakie
using Colors
using ColorSchemes
using LaTeXStrings
using DataFrames
using Statistics

# julia_colors = Colors.JULIA_LOGO_COLORS


"""
Get a pipe src and dst nodes.
"""
function pipe_nodes(network::Network, pipe::Int)
    src = findall(x -> x == -1, network.A[pipe, :])[1]
    dst = findall(x -> x == 1, network.A[pipe, :])[1]
    return src, dst
end


"""
Get normalized x and y positions for plotting.
"""
function get_graphing_x_y(network)
    pos_x = (2 * network.X_coord .- (minimum(network.X_coord) + maximum(network.X_coord))) ./ (maximum(network.X_coord) - minimum(network.X_coord))
    pos_y = (-2 * network.Y_coord .- (maximum(network.Y_coord) + minimum(network.Y_coord))) ./ (minimum(network.Y_coord) - maximum(network.Y_coord))
    return pos_x, pos_y
end


"""
Get the network as a Graphs.jl object.
"""
function network_graph(network::Network; directed=false)
    if directed
        return SimpleDiGraph(get_directed_adjacency_matrix(network))
    else
        return SimpleGraph(get_adjacency_matrix(network))
    end

end

function get_adjacency_matrix(network::Network, w=nothing)
    if w === nothing
        w = ones(network.n_l,)
    else
        error("Not supported")
    end

    iA = Int64[]
    jA = Int64[]
    vA = Float64[]


    for l in 1:network.n_l
        node_in = findall(network.A[l, :] .> 0)
        node_out = findall(network.A[l, :] .< 0)
        append!(iA, node_in)
        append!(jA, node_out)
        append!(vA, l)

        append!(iA, node_out)
        append!(jA, node_in)
        append!(vA, l)
    end

    return sparse(iA, jA, vA, size(network.A, 2), size(network.A, 2), max)
end

function get_directed_adjacency_matrix(network::Network, w=nothing)
    if w === nothing
        w = ones(network.n_l,)
    else
        error("Not supported")
    end
    A = hcat(network.A12, network.A10_res, network.A10_tank)
    iA = Int64[]
    jA = Int64[]
    vA = Float64[]
    num_junctions = network.n_n
    adj = zeros(num_junctions, num_junctions)

    for l in 1:network.n_l
        node_in = findall(A[l, :] .> 0)
        node_out = findall(A[l, :] .< 0)
        @debug node_in, node_out
        if length(node_in) != 1 || length(node_out) != 1
            error("$node_in, $node_out")
        end
        node_in, node_out = node_in[1], node_out[1]
        adj[node_in, node_out] = 1
        adj[node_out, node_in] = -1
    end

    return sparse(adj)
end




"""
Function to plot network layout with control elements (e.g. pumps and valves), reservoirs (inlets), and tanks
"""
function plot_network_layout(network; links=true, junctions=true, reservoirs=true, tanks=true, pumps=false, prvs=false, legend=false, fig_size=(500, 400), legend_pos=:lt, save_fig=false)

    pos_x, pos_y = get_graphing_x_y(network)


    # CairoMakie.activate!(type="svg") #hide

    g = network_graph(network)
    pos_x, pos_y = get_graphing_x_y(network)
    layout = Point.(zip(pos_x, pos_y))

    # node markers
    node_marker = []
    for idx in vertices(g)
        if idx in network.reservoir_idx
            push!(node_marker, :rect)
        elseif idx in network.tank_idx
            push!(node_marker, :hexagon)
        else # junction
            push!(node_marker, :circle)
        end
    end

    f = Figure(size=fig_size)
    ax = Axis(f[1, 1])

    # plot main graph
    graphplot!(
        g;
        layout=layout,
        edge_width = 1,
        edge_color=:black,
        node_color=:black,
        node_size=10,
        node_marker=:circle,
        );
    

    # plot reservoir nodes
    if reservoirs
        for reservoir in network.reservoir_idx
            if reservoir == network.reservoir_idx[end]
                Makie.scatter!(
                    pos_x[reservoir], pos_y[reservoir],
                    marker=:rect,
                    color=:royalblue4,
                    strokewidth=1,
                    strokecolor=:white,
                    markersize=20,
                    label = "Reservoir"
                    );
            else
                Makie.scatter!(
                    pos_x[reservoir], pos_y[reservoir],
                    marker=:rect,
                    color=:royalblue4,
                    strokewidth=1,
                    strokecolor=:white,
                    markersize=20
                    );
            end
        end
    end

    # plot tank nodes
    if tanks
        for tank in network.tank_idx
            if tank == network.tank_idx[end]
                Makie.scatter!(
                    pos_x[tank], pos_y[tank],
                    marker=:hexagon,
                    color=:steelblue1,
                    strokewidth=1,
                    strokecolor=:white,
                    markersize=20,
                    label = "Tank"
                    );
            else
                Makie.scatter!(
                    pos_x[tank], pos_y[tank],
                    marker=:hexagon,
                    color=:steelblue1,
                    strokewidth=1,
                    strokecolor=:white,
                    markersize=20
                    );
            end
        end
    end


    # plot prv links
    if prvs
        prv_idx = [network.link_name_to_idx[parse(Int, name)] for name in network.prv_names]
        for prv in prv_idx
            src, dst = pipe_nodes(network, prv)
            src_x, src_y = pos_x[src], pos_y[src]
            dst_x, dst_y = pos_x[dst], pos_y[dst]
            half_x = (src_x + dst_x) / 2
            half_y = (src_y + dst_y) / 2
            if prv == prv_idx[end]
                Makie.scatter!(
                    half_x, half_y,
                    markersize=18,
                    marker=:diamond,
                    color=:gray,
                    strokewidth=1,
                    strokecolor=:white,
                    label = "PRV"
                    );
            else
                Makie.scatter!(
                    half_x, half_y,
                    markersize=18,
                    marker=:diamond,
                    color=:gray,
                    strokewidth=1,
                    strokecolor=:white
                    );
            end
        end
    end

    if pumps
        for pump in network.pump_idx
            src, dst = pipe_nodes(network, pump)
            src_x, src_y = pos_x[src], pos_y[src]
            dst_x, dst_y = pos_x[dst], pos_y[dst]
            half_x = (src_x + dst_x) / 2
            half_y = (src_y + dst_y) / 2
            if pump == network.pump_idx[end]
                Makie.scatter!(
                    half_x, half_y,
                    markersize=18,
                    marker=:rtriangle,
                    color=:green,
                    strokewidth=1,
                    strokecolor=:white,
                    label = "Pump"
                    );
            else
                Makie.scatter!(
                    half_x, half_y,
                    markersize=18,
                    marker=:rtriangle,
                    color=:green,
                    strokewidth=1,
                    strokecolor=:white
                    );
            end
        end
    end
    
    hidedecorations!(ax); hidespines!(ax); ax.aspect = DataAspect(); 
    if legend
        axislegend(ax, position=legend_pos, groupgap=25, labelsize=16, show=true, framevisible=false, patchlabelgap=10)
    end

    if save_fig
        save(pwd() * "/plots/" * network.name * "_layout.pdf", f)
    end

    return f

end







"""
Function to plot network simulation results
"""
function plot_network_sim(network, state_df, state_to_plot; links=true, junctions=true, reservoirs=true, tanks=true, pumps=false, prvs=false, legend=false, fig_size=(500, 400), legend_pos=:lt, time_to_plot=1, save_fig=false)

    pos_x, pos_y = get_graphing_x_y(network)

    # CairoMakie.activate!(type="svg")
    # set_theme!(size=fig_size)

    g = network_graph(network)
    pos_x, pos_y = get_graphing_x_y(network)
    layout = Point.(zip(pos_x, pos_y))

    # edge sortig
    edge_sorted = []
    for (idx, edge) ∈ enumerate(edges(g))
        i = findall(row -> row[src(edge)] == -1 && row[dst(edge)] == 1, eachrow(network.A))
        if isempty(i)
            i = findall(row -> row[src(edge)] == 1 && row[dst(edge)] == -1, eachrow(network.A))
        end 
        i = i[1]
        push!(edge_sorted, i)
    end

    # oranize plotting data
    node_vals = []
    edge_vals = []
    if state_to_plot ∈ ["pressure", "head", "demand", "chlorine", "age", "trace"] # node values
        edge_size = 1
        node_vals = [state_df[time_to_plot, string(network.node_names[idx])][1] for idx in 1:network.n_n]
        edge_vals = :black
        if state_to_plot ∈ ["chlorine", "age"]
            cmin = round(minimum(node_vals), digits=1)
            cmax = round(maximum(node_vals), digits=1)
        else
            cmin = round(minimum(node_vals), digits=-1)
            cmax = round(maximum(node_vals), digits=-1)
        end
        cmap = :Spectral
    elseif state_to_plot ∈ ["flow", "velocity"] # edge values
        edge_size = 2
        node_vals = :black
        edge_vals = [abs(state_df[time_to_plot, string(network.link_names[idx])][1]) for idx in edge_sorted]
        if state_to_plot ∈ ["velocity"]
            cmin = round(minimum(edge_vals), digits=1)
            cmax = round(maximum(edge_vals), digits=1)
        else
            cmin = round(minimum(edge_vals), digits=-1)
            cmax = round(maximum(edge_vals), digits=-1)
        end
        # cmap = cgrad(:RdYlBu, 5, categorical = true)
        cmap = :Spectral
    end

    # node markers
    node_marker = []
    for idx in vertices(g)
        if idx in network.reservoir_idx
            push!(node_marker, :rect)
        elseif idx in network.tank_idx
            push!(node_marker, :hexagon)
        else # junction
            push!(node_marker, :circle)
        end
    end

    # node size
    node_size = []
    for idx in vertices(g)
        if idx in network.reservoir_idx
            push!(node_size, 20)
        elseif idx in network.tank_idx
            push!(node_size, 20)
        else # junction
            if state_to_plot ∉ ["pressure", "head", "demand", "chlorine", "age", "trace"]
                push!(node_size, 0)
            else
                push!(node_size, 10)
            end
        end
    end

    f = Figure(size=fig_size)

    # plot title 
    f[1, 1] = title = Label(f, state_to_plot * " @ t = " *string(state_df[time_to_plot, 1]) * " h", fontsize=16)
    title.tellwidth = false

    # plot main graph
    f[2, 1] = ax = Axis(f)
    p = graphplot!(ax, 
        g;
        layout=layout,
        edge_width=edge_size,
        node_size=node_size,
        node_marker=node_marker,
        edge_attr=(colorrange=(cmin,cmax), color=edge_vals, colormap=cmap),
        node_attr=(colorrange=(cmin,cmax), color=node_vals, colormap=cmap),
        );
    hidedecorations!(ax); hidespines!(ax); ax.aspect = DataAspect(); 

    # plot prv links
    if prvs
        prv_idx = [network.link_name_to_idx[parse(Int, name)] for name in network.prv_names]
        for prv in prv_idx
            src, dst = pipe_nodes(network, prv)
            src_x, src_y = pos_x[src], pos_y[src]
            dst_x, dst_y = pos_x[dst], pos_y[dst]
            half_x = (src_x + dst_x) / 2
            half_y = (src_y + dst_y) / 2
            Makie.scatter!(ax, 
                half_x, half_y,
                markersize=18,
                marker=:diamond,
                color=:black,
                strokewidth=1,
                strokecolor=:white,
                );
        end
    end

    if pumps
        for pump in network.pump_idx
            src, dst = pipe_nodes(network, pump)
            src_x, src_y = pos_x[src], pos_y[src]
            dst_x, dst_y = pos_x[dst], pos_y[dst]
            half_x = (src_x + dst_x) / 2
            half_y = (src_y + dst_y) / 2
            Makie.scatter!(ax, 
                half_x, half_y,
                markersize=18,
                marker=:rtriangle,
                color=:black,
                strokewidth=1,
                strokecolor=:white,
                );
        end
    end

    # plot colorbar
    if state_to_plot ∈ ["pressure", "head", "demand", "chlorine", "age", "trace"]
        if state_to_plot == "pressure"
            label = "Pressure [m]"
        elseif state_to_plot == "head"
            label = "Head [m]"
        elseif state_to_plot == "demand"
            label = "Demand [L/s]"
        elseif state_to_plot == "chlorine"
            label = "Chlorine [mg/L]"
        elseif state_to_plot == "age"
            label = "Age [h]"
        elseif state_to_plot == "trace"
            label = "Trace [%]"
        end
        f[2,2] = Colorbar(f, get_node_plot(p), label=label, ticklabelsize=14, labelsize=16)
    else
        if state_to_plot == "flow"
            label = "Flow [L/s]"
        elseif state_to_plot == "velocity"
            label = "Velocity [m/s]"
        end
        f[2,2] = Colorbar(f, get_edge_plot(p), label=label, ticklabelsize=14, labelsize=16)
    end

    if save_fig
        save(pwd() * "/plots/" * network.name * "_" * state_to_plot * "_network.pdf", f)
    end
    
    return f
end





"""
Function to plot simulation time series at select elements
"""
function plot_timeseries_sim(network, state_df, state_to_plot, elements_to_plot; fig_size=(600, 450), save_fig=false)

    if length(elements_to_plot) > 1
        ymin = minimum(describe(state_df[!, string.(elements_to_plot)]).min)
        ymax = maximum(describe(state_df[!, string.(elements_to_plot)]).max)
    elseif length(elements_to_plot) == 1
        ymin = minimum(state_df[!, string.(elements_to_plot)])
        ymax = maximum(state_df[!, string.(elements_to_plot)])
    else
        @info "No elements selected to plot."
    end
    xmax = round(maximum(state_df.timestamp), digits=0)


    if state_to_plot == "pressure"
        ylabel = "Pressure [m]"
        legend = "Nodes"
        ymin = 10 * floor(ymin / 10)
        ymax = 10 * ceil(ymax / 10)
    elseif state_to_plot == "head"
        ylabel = "Head [m]"
        legend = "Nodes"
        ymin = 10 * floor(ymin / 10)
        ymax = 10 * ceil(ymax / 10)
    elseif state_to_plot == "demand"
        ylabel = "Demand [L/s]"
        legend = "Nodes"
        ymin = 10 * floor(ymin / 10)
        ymax = 10 * ceil(ymax / 10)
    elseif state_to_plot == "chlorine"
        ylabel = "Chlorine [mg/L]"
        legend = "Nodes"
        ymin = 0.5 * floor(ymin / 0.5)
        ymax = 0.5 * ceil(ymax / 0.5)
    elseif state_to_plot == "age"
        ylabel = "Age [h]"
        legend = "Nodes"
        ymin = 20 * floor(ymin / 20)
        ymax = 20 * ceil(ymax / 20)
    elseif state_to_plot == "trace"
        ylabel = "Trace [%]"
        legend = "Nodes"
        ymin = 20 * floor(ymin / 20)
        ymax = 20 * ceil(ymax / 20)
    elseif state_to_plot == "flow"
        ylabel = "Flow [L/s]"
        legend = "Links"
        ymin = 100 * floor(ymin / 100)
        ymax = 100 * ceil(ymax / 100)
    elseif state_to_plot == "velocity"
        ylabel = "Velocity [m/s]"
        legend = "Links"
        if ymax > 1
            ymin = 0.5 * floor(ymin / 0.5)
            ymax = 0.5 * ceil(ymax / 0.5)
        else
            ymin = 0.2 * floor(ymin / 0.2)
            ymax = 0.2 * ceil(ymax / 0.2)
        end
    end

    f = Figure(size=fig_size)
    ax = Axis(f[1, 1],
        xlabel = "Time [h]",
        ylabel = ylabel,
        xlabelsize = 16,
        ylabelsize = 16,
        xticks = 0:xmax/4:xmax,
        # yticks= ymin:(ymax-ymin)/4:ymax,

    )
    ylims!(low=ymin, high=ymax)
    xlims!(low=0, high=xmax)
    x = state_df.timestamp
    for element in elements_to_plot
        lines!(ax, x, state_df[!, string(element)], label=string(element), linewidth = 1.5)
    end

    f[1, 2] = axislegend(legend, labelsize=14, framevisible=false, position=:rt)

    if save_fig
        save(pwd() * "/plots/" * network.name * "_" * state_to_plot * "_timeseries.pdf", f)
    end

    return f


end






"""
Function to plot comparison between EPANET and water quality solver results
"""
function plot_wq_solver_comparison(network, state_df, c, node_to_plot, disc_method, Δt, Δk; fig_size=(700, 350), save_fig=true)

    # organize wq_solver results
    c_r = c[1:network.n_r, :]
    c_j = c[network.n_r+1:network.n_r+network.n_j, :]
    c_tk = c[network.n_r+network.n_j+1:network.n_r+network.n_j+network.n_tk, :]

    if disc_method == "explicit-central"
        disc_method = "Explicit Central"
    elseif disc_method == "implicit-upwind"
        disc_method = "Implicit Upwind"
    elseif disc_method == "explicit-upwind"
        disc_method = "Explicit Upwind"
    elseif disc_method == "implicit-central"
        disc_method = "Implicit Central"
    end

    ylabel = "Chlorine [mg/L]"
    ymin = minimum(state_df[!, string.(node_to_plot)])
    ymin = 0.5 * floor(ymin / 0.5)
    ymax = 1.1 * maximum(state_df[!, string.(node_to_plot)])
    ymax = 0.5 * ceil(ymax / 0.5)
    xmax = round(maximum(state_df.timestamp), digits=0)


    f = Figure(size=fig_size)
    ax = Axis(f[1, 1],
        title = "Node " * string(node_to_plot),
        # titlefont = "normal",
        xlabel = "Time [h]",
        ylabel = ylabel,
        titlesize = 16,
        xlabelsize = 16,
        ylabelsize = 16,
        xticks = 0:xmax/4:xmax,
        # yticks= ymin:(ymax-ymin)/4:ymax,
    )
    ylims!(low=ymin, high=ymax)
    xlims!(low=0, high=xmax)
    x = state_df.timestamp

    # EPANET solver results
    epanet = lines!(ax, x, state_df[!, string(node_to_plot)], label="EPANET", linewidth=1.5)

    # wq_solver results
    node_idx = network.node_name_to_idx[node_to_plot]
    if node_idx in network.reservoir_idx
        plot_idx = findfirst(x -> x == node_idx, network.reservoir_idx)
        y = c_r[plot_idx, :]
    elseif node_idx in network.junction_idx
        plot_idx = findfirst(x -> x == node_idx, network.junction_idx)
        y = c_j[plot_idx, :]
    else
        plot_idx = findfirst(x -> x == node_idx, network.tank_idx)
        y = c_tk[plot_idx, :]
    end
    wq_solver = lines!(ax, x, y, label=disc_method, linewidth=1.5)

    dummy = lines!(x, y, color=:white, linewidth=0.0)


    # add legend
    f[1, 2] = axislegend(ax, [epanet, wq_solver], ["EPANET", disc_method], "Δt="*string(Δt)*" s, Δk="*string(Δk)*" s", position = :rt, labelsize=14, framevisible=false, titlefont="normal")
    # axislegend(ax, [epanet, wq_solver, dummy], ["EPANET", disc_method, "(Δt="*string(Δt)*", Δk="*string(Δk)*" s)"], position = :rt, labelsize=14, framevisible=false)
    # f[1, 2] = axislegend(legend_title, labelsize=14, framevisible=false, position=:rt, fontstyle="normal")

    if save_fig
        save(pwd() * "/plots/" * network.name * "_node_" * string(node_to_plot) * "_solver_comparison_Δt_" * string(Δt) * "_Δk_" * string(Δk) * ".pdf", f)
    end

    return f


end