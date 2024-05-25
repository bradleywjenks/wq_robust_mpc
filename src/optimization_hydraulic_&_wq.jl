"""
Collection of functions for optimizing hydraulic and water quality states
"""

using InlineStrings
using Parameters
using AutomaticDocstrings
using PyCall
using DataFrames
using LinearAlgebra
using SparseArrays
using JuMP
using Ipopt
using Gurobi
using NLsolve



@with_kw mutable struct OptParams
    obj_type::String
    disc_method::String
    Qmin::Matrix{Float64}
    Qmax::Matrix{Float64}
    pmin::Float64
    Hmin::Matrix{Float64}
    Hmax::Matrix{Float64}
    Xmin_r::Matrix{Float64}
    Xmax_r::Matrix{Float64}
    Xmin_j::Matrix{Float64}
    Xmax_j::Matrix{Float64}
    Xmin_tk::Matrix{Float64}
    Xmax_tk::Matrix{Float64}
    Umin::Matrix{Float64}
    Umax::Matrix{Float64}
    b_loc::Union{Vector{Int64}, Nothing}
    m_loc::Union{Vector{Int64}, Nothing}
    a::Union{Vector{Float64}, Nothing}
    b::Union{Vector{Float64}, Nothing}
    Δx_p::Union{Vector{Float64}, Nothing}
    s_p::Union{Vector{Int64}, Nothing}
    T::Int64
    k_t::Vector{Int64}
    kb::Float64
end

Base.copy(o::OptParams) = OptParams(
    obj_type=deepcopy(o.obj_type),
    disc_method=deepcopy(o.disc_method),
    Qmin=deepcopy(o.Qmin),
    Qmax=deepcopy(o.Qmax),
    pmin=deepcopy(o.pmin),
    Hmin=deepcopy(o.Hmin),
    Hmax=deepcopy(o.Hmax),
    Xmin_r=deepcopy(o.Xmin_r),
    Xmax_r=deepcopy(o.Xmax_r),
    Xmin_j=deepcopy(o.Xmin_j),
    Xmax_j=deepcopy(o.Xmax_j),
    Xmin_tk=deepcopy(o.Xmin_tk),   
    Xmax_tk=deepcopy(o.Xmax_tk),
    Umin=deepcopy(o.Umin),
    Umax=deepcopy(o.Umax),
    b_loc=deepcopy(o.b_loc),
    m_loc=deepcopy(o.m_loc),
    a=deepcopy(o.a),
    b=deepcopy(o.b),
    Δx_p=deepcopy(o.Δx_p),
    s_p=deepcopy(o.s_p),
    T=deepcopy(o.T),
    k_t=deepcopy(o.k_t),
    kb=deepcopy(o.kb),
)




"""
Function for making optimization parameters (hydraulic and water quality), which include:
    - state and control variable bounds
    - water quality discretization parameters
    - frictional loss formula
    - optimization objective

"""
function make_prob_data(network::Network, Δt, Δk, sim_days, disc_method; pmin::Int64=15, Qmax_mul=Inf, Qmin_mul=-Inf, x_bounds=(0.5, 3), u_bounds=(0, 5), QA=false, quadratic_approx="relative", J=nothing, kb=0.5, kw=0)

    n_t = Int(get_hydraulic_time_steps(network, network.name, sim_days, Δk))
    sim_type = "hydraulic"
    sim_results = epanet_solver(network, sim_type; sim_days=sim_days, Δk=Δk)

    # h bounds at junctions
    j_demand = all(>(0), network.d, dims=2)
    Hmin = [j ∈ j_demand ? pmin + network.elev[network.junction_idx[j]] : network.elev[network.junction_idx[j]] for j in 1:network.n_j]
    Hmin = repeat(Hmin, 1, n_t)
    Hmax = repeat(maximum(network.h0, dims=2), 1, n_t)

    # q bounds across links
    Qmin = Qmin_mul * ones(network.n_l, n_t)
    Qmax = Qmax_mul * ones(network.n_l, n_t)

    # u bounds at booster locations
    b_loc, _ = get_booster_inputs(network, network.name, sim_days, Δk, Δt) # booster control locations
    Umin = u_bounds[1] * ones(length(b_loc), n_t)
    Umax = u_bounds[2] * ones(length(b_loc), n_t)

    # pump locations
    m_loc = findall(x -> x==1, network.B_pump)

    # water quality bounds at nodes
    Xmin_r = 0 * ones(network.n_r, n_t)
    Xmax_r = Inf * ones(network.n_r, n_t)
    Xmin_j = x_bounds[1] * ones(network.n_j, n_t)
    Xmax_j = x_bounds[2] * ones(network.n_j, n_t)
    Xmin_tk = x_bounds[1] * ones(network.n_tk, n_t)
    Xmax_tk = x_bounds[2] * ones(network.n_tk, n_t)

    # frictional loss model
    v_init = Matrix(getfield(sim_results, :velocity))'
    v_max = vmax(network, v_init)
    v_max = v_max[network.pipe_idx, :]
    if QA
        a, b = quadratic_app(network, v_max, quadratic_approx)
    end

    # set discretization parameters and variables
    s_p = []
    n_s = []
    vel_p = v_init[network.pipe_idx, :]
    L_p = network.L[network.pipe_idx]
    if J === nothing
        vel_p_max = maximum(vel_p, dims=2)
        s_p = vec(L_p ./ (vel_p_max .* Δt))
        s_p = floor.(Int, s_p)
        s_p[s_p .== 0] .= 1
    else
        s_p = J .* ones(network.n_p)
    end

    Δx_p = L_p ./ s_p

    # check CFL condition
    if disc_method == "explicit-central" || disc_method == "explicit-upwind"
        for k ∈ 1:n_t
            if any(Δt .>= Δx_p ./ vel_p[:, k])
                @error "CFL condition not satisfied. Please input a smaller Δt."
                return s_p
            end
        end
    end

    # model time steps
    sim_duration = round(sim_results.timestamp[end] + (sim_results.timestamp[end] - sim_results.timestamp[end-1]), digits=0) * 3600
    k_set = sim_results.timestamp .* 3600
    T = Int(sim_duration / Δt) # number of discrete time steps
    k_t = zeros(T+1)
    for t ∈ 1:T
        k_t[t] = searchsortedfirst(k_set, t*Δt) - 1
    end
    k_t[end] = k_t[1]
    k_t = Int.(k_t)



    return OptParams(
        obj_type="missing",
        disc_method=disc_method,
        Qmin=Qmin,
        Qmax=Qmax,
        pmin=pmin,
        Hmin=Hmin,
        Hmax=Hmax,
        Xmin_r=Xmin_r,
        Xmax_r=Xmax_r,
        Xmin_j=Xmin_j,
        Xmax_j=Xmax_j,
        Xmin_tk=Xmin_tk,
        Xmax_tk=Xmax_tk,
        Umin=Umin,
        Umax=Umax,
        b_loc=b_loc,
        m_loc=m_loc,
        a=a,
        b=b,
        Δx_p=Δx_p,
        s_p=s_p, 
        T=T,
        k_t=k_t,
        kb=kb,
    )
end



"""
Determine the maximum v for a network
    vmul = multipliers to use
"""
function vmax(network, v_init; vmul::Vector{<:Real}=[1, 2, 3, 4, 5, 6])
    v_init = abs.(v_init)
    v_max = zeros(size(v_init, 1), size(v_init, 2))
    v_max[v_init.<0.5] .= vmul[1]
    v_max[0.5 .≤ v_init .< 1] .= vmul[2]
    v_max[1 .≤ v_init .< 2] .= vmul[3]
    v_max[2 .≤ v_init .< 3] .= vmul[4]
    v_max[3 .≤ v_init .< 4] .= vmul[5]
    v_max[4 .≤ v_init .< 5] .= vmul[6]
    return v_max
end



"""
Get quadratic approximation parameters
"""
function quadratic_app(network, v_max, quadratic_approx; ϵ=0.1)

    a = zeros(network.n_p, 1)
    b = zeros(network.n_p, 1)
    pipe_area = (pi .* network.D[network.pipe_idx] .^ 2) ./ 4
    Q = v_max[network.pipe_idx] .* pipe_area .* 1000 # convert to L/s

    if network.simtype == "H-W"

        r = network.C[network.pipe_idx]
        n = network.nexp[network.pipe_idx][1]

        if quadratic_approx == "absolute"

            if findmax(Q) == Inf
                @error "Maximum flow value must be finite."
            end

            A1 = (Q .^ 5) ./ 5
            A2 = (Q .^ 3) ./ 3
            A3 = (Q .^ 4) ./ 4
            A4 = (Q .^ (3 + n)) ./ (3 + n)
            A5 = (Q .^ (2 + n)) ./ (2 + n)
        
            b = ((r .* (A5 .* A1 - A3 .* A4)) ./ (A2 .* A1 - A3 .^ 2))
            a = ((r .* A4) - (b .* A3)) ./ A1

        elseif quadratic_approx == "relative"

            function _k_eps(kk, ϵ)
                kk = kk[1]
                n = 1.852
                aa1 = (1 - kk .^ (5 - 2 * n)) / (5 - 2 * n)
                aa2 = (1 - kk .^ (3 - 2 * n)) / (3 - 2 * n)
                aa3 = (1 - kk .^ (4 - 2 * n)) / (4 - 2 * n)
                aa4 = (1 - kk .^ (3 - n)) / (3 - n)
                aa5 = (1 - kk .^ (2 - n)) / (2 - n)
                llb = (aa5 .* aa1 - aa4 .* aa3) ./ (aa2 .* aa1 - aa3 .^ 2)
                lla = (aa4 - llb .* aa3) ./ aa1
                z = lla^(n - 1) * llb^(2 - n) + (ϵ - 1) * (2 - n) * ((n - 1) / (2 - n))^(n - 1)
                return z
            end

            k0 = 1e-6
            k_eps(kk) = _k_eps(kk, ϵ)
        
            k = nlsolve(k_eps, [k0], autodiff=:forward).zero[1]
            a1 = (1 - k .^ (5 - 2 * n)) / (5 - 2 * n)
            a2 = (1 - k .^ (3 - 2 * n)) / (3 - 2 * n)
            a3 = (1 - k .^ (4 - 2 * n)) / (4 - 2 * n)
            a4 = (1 - k .^ (3 - n)) / (3 - n)
            a5 = (1 - k .^ (2 - n)) / (2 - n)
            lb = (a5 .* a1 - a4 .* a3) ./ (a2 .* a1 - a3 .^ 2)
            la = (a4 - lb .* a3) ./ a1
            a = r .* Q .^ (n - 2) * la
            b = r .* Q .^ (n - 1) * lb
            @assert all(0 .≤ a)
            @assert all(0 .≤ b)
        end

        return a, b

    end

end





"""
Main function for solving the joint hydraulic and water quality optimization problem
"""
function optimize_hydraulic_wq(network::Network, opt_data::OptParams)

    ##### SET OPTIMIZATION PARAMETERS #####

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
    h0 = network.h0
    d = network.d
    A12 = network.A12
    A10_res = network.A10_res
    A10_tank = network.A10_tank
    A_inc = hcat(A12, A10_res, A10_tank)
    elev_tk = network.elev[tank_idx]
    area_tk = network.tank_area

    # unload optimization parameters
    disc_method = opt_data.disc_method
    Qmin = opt_data.Qmin
    Qmax = opt_data.Qmax
    Hmin = opt_data.Hmin
    Hmax = opt_data.Hmax
    Xmin_r = opt_data.Xmin_r
    Xmax_r = opt_data.Xmax_r
    Xmin_j = opt_data.Xmin_j
    Xmax_j = opt_data.Xmax_j
    Xmin_tk = opt_data.Xmin_tk
    Xmax_tk = opt_data.Xmax_tk
    Umin = opt_data.Umin
    Umax = opt_data.Umax
    b_loc = opt_data.b_loc
    m_loc = opt_data.m_loc
    a = opt_data.a
    b = opt_data.b
    Δx_p = opt_data.Δx_p
    s_p = opt_data.s_p
    n_s = sum(s_p)
    kb = (opt_data.kb/3600/24) # units = 1/second

    # boundary data
    h0 = network.h0

    # simulation times
    T = opt_data.T
    k_t = opt_data.k_t


    ##### BUILD OPTIMIZATION MODEL #####



    return nothing, nothing, nothing, nothing

end