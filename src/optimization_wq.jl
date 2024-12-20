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
Main function for optimizing water quality (with known hydraulics)
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
    Sh = zeros(n_p, n_t)
    kf = zeros(n_p, n_t)
    for i ∈ 1:n_p
        for k ∈ 1:n_t
            if Re[i, k] < 2400
                Sh[i, k] = 3.65 + (0.0668 * (D_p[i]/L_p[i]) * Re[i, k] * Sc) /  (1 + 0.04 * ((D_p[i]/L_p[i]) * Re[i, k] * Sc)^(2/3))
            else
                Sh[i, k] = 0.0149 * Re[i, k]^0.88 * Sc^(1/3)
            end
        end
        kf[i, :] .= Sh[i, :] .* (D_m / D_p[i])
    end

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
    # if any(s_p .< 0.75)
    #     @error "At least one pipe has discretization step Δx > pipe length. Please input a smaller Δt."
    #     return
    # end
    s_p = floor.(Int, s_p)
    s_p[s_p .== 0] .= 1
    n_s = sum(s_p)
    Δx_p = L_p ./ s_p
    λ_p = vel_p ./ repeat(Δx_p, 1, n_t) .* Δt
    λ_p = λ_p .* qdir[pipe_idx, :]

    # check CFL condition
    if disc_method == "explicit-central" || disc_method == "explicit-upwind"
        for k ∈ 1:n_t
            if any(Δt .>= Δx_p ./ vel_p[:, k])
                @error "CFL condition not satisfied. Please input a smaller Δt."
                return s_p
            end
        end
    end

    # simulation times
    T = round(sim_results.timestamp[end] + (sim_results.timestamp[end] - sim_results.timestamp[end-1]), digits=0) * 3600
    T_k = Int(T / Δt) # number of discrete time steps
    k_t = zeros(1, T_k+1)
    for t ∈ 1:T_k
        k_t[t] = searchsortedfirst(k_set, t*Δt) - 1
    end
    k_t[end] = k_t[1]
    k_t = Int.(k_t)
    print("This is k_t ")
    println(k_t)


    
    ##### BUILD OPTIMIZATION MODEL #####

    ### GUROBI optimizer
    model = Model(Gurobi.Optimizer)
    # set_optimizer_attribute(model,"Method", 2)
    # set_optimizer_attribute(model,"Presolve", 0)
    # set_optimizer_attribute(model,"Crossover", 0)
    # set_optimizer_attribute(model,"NumericFocus", 3)
    # set_optimizer_attribute(model,"NonConvex", 2)
    # set_silent(model)

    ### define variables
    # @variable(model, x_bounds[1] ≤ c_r[i=1:n_r] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_r[i=1:n_r, t=1:T_k+1] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_j[i=1:n_j, t=1:T_k+1] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_tk[i=1:n_tk, t=1:T_k+1] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_m[i=1:n_m, t=1:T_k+1] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_v[i=1:n_v, t=1:T_k+1] ≤ x_bounds[2])
    @variable(model, x_bounds[1] ≤ c_p[i=1:n_s, t=1:T_k+1] ≤ x_bounds[2])
    @variable(model, u_bounds[1] ≤ u[i=1:n_j, t=1:n_t] ≤ u_bounds[2])


    ### define constraints

    # initial and boundary (reservoir) conditions
    @constraint(model, c_r .== repeat(source_cl, 1, T_k+1))
    @constraint(model, c_j[:, 1] .== x0)
    @constraint(model, c_tk[:, 1] .== x0)
    @constraint(model, c_m[:, 1] .== x0)
    @constraint(model, c_v[:, 1] .== x0)
    @constraint(model, c_p[:, 1] .== x0)

    # booster locations
    b_idx = findall(x -> x ∈ b_loc, junction_idx)
    not_b_idx = setdiff(1:length(junction_idx), b_idx)
    @constraint(model, u[not_b_idx, :] .== 0)
    # @constraint(model, u .== 0)

    # junction mass balance
    ϵ_reg = 1e-3
    @constraint(model, junction_balance[i=1:n_j, t=2:T_k+1],
        c_j[i, t] == (
            sum(
                (q[j, k_t[t-1]] + ϵ_reg) * (
                    j in pipe_idx ? 
                        c_p[
                            qdir[j, k_t[t-1]] == 1 ? 
                            sum(s_p[1:findfirst(x -> x == j, pipe_idx)]) :
                            sum(s_p[1:findfirst(x -> x == j, pipe_idx) - 1]) + 1, t
                        ] :
                    j in pump_idx ? 
                        c_m[findfirst(x -> x == j, pump_idx), t] :
                        c_v[findfirst(x -> x == j, valve_idx), t]
                ) for j in findall(x -> x == 1, A_inc[:, junction_idx[i], k_t[t-1]])
            ) / (
                d[i, k_t[t-1]] + sum(q[j, k_t[t-1]] for j in findall(x -> x == -1, A_inc[:, junction_idx[i], k_t[t-1]])) + ϵ_reg
            )
        ) + u[i, k_t[t-1]]
    )

    # tank mass balance
    @constraint(model, tank_balance[i=1:n_tk, t=2:T_k+1],
        c_tk[i, t] == (
            c_tk[i, t-1] * V_tk[i, k_t[t-1]] -
            (
                c_tk[i, t-1] * Δt * sum(
                    q[j, k_t[t-1]] for j in findall(x -> x == -1, A_inc[:, tank_idx[i], k_t[t-1]])
                )
            ) +
            (
                sum(
                    q[j, k_t[t-1]] * Δt * (
                        j in pipe_idx ? 
                            c_p[
                                qdir[j, k_t[t-1]] == 1 ? 
                                sum(s_p[1:findfirst(x -> x == j, pipe_idx)]) :
                                sum(s_p[1:findfirst(x -> x == j, pipe_idx) - 1]) + 1, t-1
                            ] :
                        (j in pump_idx ? 
                            c_m[findfirst(x -> x == j, pump_idx), t-1] :
                            c_v[findfirst(x -> x == j, valve_idx), t-1]
                        )
                    ) for j in findall(x -> x == 1, A_inc[:, tank_idx[i], k_t[t-1]])
                )
            ) +
            (
                -1 * c_tk[i, t-1] * kb * V_tk[i, k_t[t-1]] * Δt
            )
        ) / V_tk[i, k_t[t]]
    )

    # pump mass balance
    @constraint(model, pump_balance[i=1:n_m, t=2:T_k+1],
        c_m[i, t] == begin
            node_idx = findall(x -> x == -1, A_inc[pump_idx[i], :, k_t[t-1]])[1]
            c_up = node_idx ∈ reservoir_idx ? 
                c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                node_idx ∈ junction_idx ? 
                    c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                    c_tk[findfirst(x -> x == node_idx, tank_idx), t]
            c_up
        end
    )

    # valve mass balance
    @constraint(model, valve_balance[i=1:n_v, t=2:T_k+1],
        c_v[i, t] == begin
            node_idx = findall(x -> x == -1, A_inc[valve_idx[i], :, k_t[t-1]])[1]
            c_up = node_idx ∈ reservoir_idx ? 
                c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                node_idx ∈ junction_idx ? 
                    c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                    c_tk[findfirst(x -> x == node_idx, tank_idx), t]
            c_up
        end
    )


    # pipe segment transport
    s_p_end = cumsum(s_p, dims=1)
    s_p_start = s_p_end .- s_p .+ 1

    if disc_method == "explicit-central"
        @constraint(model, pipe_transport[i=1:n_s, t=2:T_k+1], 
            c_p[i, t] .== begin
                p = findlast(x -> x <= i, s_p_start)[1]
                λ_i = λ_p[p, k_t[t-1]] 
                kf_i = kf[p, k_t[t-1]]
                D_i = D_p[p]
                k_i = kb + ((4 * kw * kf_i) / (D_i * (kw + kf_i)))
                
                c_start = i ∈ s_p_start ? begin
                    idx = findfirst(x -> x == i, s_p_start)
                    node_idx = findall(x -> x == -1, A_inc_0[pipe_idx[idx], :, 1])[1]
                    node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t-1] :
                    node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t-1] :
                    c_tk[findfirst(x -> x == node_idx, tank_idx), t-1]
                end : nothing

                c_end = i ∈ s_p_end ? begin
                    idx = findfirst(x -> x == i, s_p_end)
                    node_idx = findall(x -> x == 1, A_inc_0[pipe_idx[idx], :, 1])[1]
                    node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t-1] :
                    node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t-1] :
                    c_tk[findfirst(x -> x == node_idx, tank_idx), t-1]
                end : nothing

                i ∈ s_p_start ?
                    0.5 * λ_i * (1 + λ_i) * c_start + (1 - abs(λ_i)^2) * c_p[i, t-1] - 0.5 * λ_i * (1 - λ_i) * c_p[i+1, t-1] - c_p[i, t-1] * k_i * Δt :
                i ∈ s_p_end ?
                    0.5 * λ_i * (1 + λ_i) * c_p[i-1, t-1] + (1 - abs(λ_i)^2) * c_p[i, t-1] - 0.5 * λ_i * (1 - λ_i) * c_end - c_p[i, t-1] * k_i * Δt :
                    0.5 * λ_i * (1 + λ_i) * c_p[i-1, t-1] + (1 - abs(λ_i)^2) * c_p[i, t-1] - 0.5 * λ_i * (1 - λ_i) * c_p[i+1, t-1] - c_p[i, t-1] * k_i * Δt
            end
        )

    elseif disc_method == "implicit-upwind"
        @constraint(model, pipe_transport[i=1:n_s, t=2:T_k+1], 
            c_p[i, t] .== begin
                p = findlast(x -> x <= i, s_p_start)[1]
                qdir_i = qdir[p, k_t[t-1]]
                λ_i = abs(λ_p[p, k_t[t-1]])
                kf_i = kf[p, k_t[t-1]]
                D_i = D_p[p]
                k_i = kb + ((4 * kw * kf_i) / (D_i * (kw + kf_i)))
                
                c_up = i ∈ s_p_start && i ∉ s_p_end ? begin
                    qdir_i == 1 ? begin
                        idx = findfirst(x -> x == i, s_p_start)
                        node_idx = findall(x -> x == -1, A_inc_0[pipe_idx[idx], :, 1])[1]
                        node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t]
                    end : c_p[i+1, t] 
                end : i ∉ s_p_start && i ∈ s_p_end ? begin
                    qdir_i == -1 ? begin
                        idx = findfirst(x -> x == i, s_p_end)
                        node_idx = findall(x -> x == 1, A_inc_0[pipe_idx[idx], :, 1])[1]
                        node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t]
                    end : c_p[i-1, t]
                end : i ∈ s_p_start && i ∈ s_p_end ? begin
                    qdir_i == 1 ? begin
                        idx = findfirst(x -> x == i, s_p_start)
                        node_idx = findall(x -> x == -1, A_inc_0[pipe_idx[idx], :, 1])[1]
                        node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t]
                    end : qdir_i == -1 ? begin
                        idx = findfirst(x -> x == i, s_p_end)
                        node_idx = findall(x -> x == 1, A_inc_0[pipe_idx[idx], :, 1])[1]
                        node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t]
                    end : nothing
                end : qdir_i == 1 ? c_p[i-1, t] : c_p[i+1, t]

                (c_p[i, t-1] * (1 - k_i * Δt) + λ_i * c_up) / (1 + λ_i)
 
            end
        )
    
    else
        @error "Discretization method has not been implemented yet."
        return
    end

    ### define objective function

    ### minimize booster cost
    ρ = 1e-3

    @objective(model, Min, ρ * sum(u[i, k] * sum(q[j, k] for j in findall(x -> x == -1, A_inc[:, junction_idx[i], k])) for i ∈ 1:n_j, k ∈ 1:n_t))
    # @objective(model, Min, sum(u[i, t] for i in 1:n_j, t in 1:n_t))

    ### solve optimization problem
    optimize!(model)



    return value.(c_r), value.(c_j), value.(c_tk), value.(c_m), value.(c_v), value.(c_p), value.(u)
    # return nothing, nothing
    # return s_p_end, s_p_start


end






