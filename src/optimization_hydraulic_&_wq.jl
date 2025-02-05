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
using SCIP
using NLsolve
using Polynomials
using LinearAlgebra


@with_kw mutable struct OptParams
    obj_type::String
    disc_method::String
    Qmin::Union{Matrix{Float64}, Vector{Float64}}
    Qmax::Union{Matrix{Float64}, Vector{Float64}}
    pmin::Float64
    Hmin_j::Union{Matrix{Float64}, Vector{Float64}}
    Hmax_j::Union{Matrix{Float64}, Vector{Float64}}
    Hmin_tk::Union{Matrix{Float64}, Vector{Float64}}
    Hmax_tk::Union{Matrix{Float64}, Vector{Float64}}
    tk_init::Vector{Float64}
    Xmin_r::Union{Matrix{Float64}, Vector{Float64}}
    Xmax_r::Union{Matrix{Float64}, Vector{Float64}}
    Xmin_j::Union{Matrix{Float64}, Vector{Float64}}
    Xmax_j::Union{Matrix{Float64}, Vector{Float64}}
    Xmin_tk::Union{Matrix{Float64}, Vector{Float64}}
    Xmax_tk::Union{Matrix{Float64}, Vector{Float64}}
    Umin_b::Union{Matrix{Float64}, Vector{Float64}}
    Umax_b::Union{Matrix{Float64}, Vector{Float64}}
    b_loc::Union{Vector{Int64}, Nothing}
    m_loc::Union{Vector{Int64}, Nothing}
    a::Union{Vector{Float64}, Matrix{Float64}, Nothing}
    b::Union{Vector{Float64}, Matrix{Float64}, Nothing}
    Δx_p::Union{Vector{Float64}, Nothing}
    s_p::Union{Vector{Int64}, Nothing}
    sim_days::Int64
    T::Int64
    Δt::Int64
    Δk::Int64
    k_t::Vector{Int64}
    kb::Float64
    QA::Bool
    θmin::Union{Matrix{Float64}, Vector{Float64}}
    θmax::Union{Matrix{Float64}, Vector{Float64}}
    x_wq_0::Float64
    max_pump_switch::Int64 
    c_elec::Union{Matrix{Float64}, Vector{Float64}}
end

Base.copy(o::OptParams) = OptParams(
    obj_type=deepcopy(o.obj_type),
    disc_method=deepcopy(o.disc_method),
    Qmin=deepcopy(o.Qmin),
    Qmax=deepcopy(o.Qmax),
    pmin=deepcopy(o.pmin),
    Hmin_j=deepcopy(o.Hmin),
    Hmax_j=deepcopy(o.Hmax),
    Hmin_tk=deepcopy(o.Hmin_tk),
    Hmax_tk=deepcopy(o.Hmax_tk),
    tk_init=deepcopy(o.tk_init),
    Xmin_r=deepcopy(o.Xmin_r),
    Xmax_r=deepcopy(o.Xmax_r),
    Xmin_j=deepcopy(o.Xmin_j),
    Xmax_j=deepcopy(o.Xmax_j),
    Xmin_tk=deepcopy(o.Xmin_tk),   
    Xmax_tk=deepcopy(o.Xmax_tk),
    Umin_b=deepcopy(o.Umin),
    Umax_b=deepcopy(o.Umax),
    b_loc=deepcopy(o.b_loc),
    m_loc=deepcopy(o.m_loc),
    a=deepcopy(o.a),
    b=deepcopy(o.b),
    Δx_p=deepcopy(o.Δx_p),
    s_p=deepcopy(o.s_p),
    sim_days=deepcopy(o.sim_days),
    T=deepcopy(o.T),
    k_t=deepcopy(o.k_t),
    Δt=deepcopy(o.Δt),
    Δk=deepcopy(o.Δk),
    kb=deepcopy(o.kb),
    QA=deepcopy(o.QA),
    θmin=deepcopy(o.θmin),
    θmax=deepcopy(o.θmax),
    x_wq_0=deepcopy(o.x_wq_0),
    max_pump_switch=deepcopy(o.max_pump_switch),
    c_elec=deepcopy(o.c_elec)
)

@with_kw mutable struct OptResults
    q::Matrix{Float64}
    h_j::Matrix{Float64}
    h_tk::Union{Nothing, Matrix{Float64}}
    u_m::Union{Nothing, Matrix{Float64}}
    q⁺::Matrix{Float64}
    q⁻::Matrix{Float64}
    s::Matrix{Float64}
    θ::Matrix{Float64}
    θ⁺::Matrix{Float64}
    θ⁻::Matrix{Float64}
    z::Union{Nothing, Matrix{Float64}}
end

Base.copy(res::OptResults) = OptResults(
    q=deepcopy(res.q),
    h_j=deepcopy(res.h_j),
    h_tk=deepcopy(res.h_tk),
    u_m=deepcopy(res.u_m),
    q⁺=deepcopy(res.q⁺),
    q⁻=deepcopy(res.q⁻),
    s=deepcopy(res.s),
    θ=deepcopy(res.θ),
    θ⁺=deepcopy(res.θ⁺),
    θ⁻=deepcopy(res.θ⁻),
    z=deepcopy(res.z)
)




"""
Function for making optimization parameters (hydraulic and water quality), which include:
    - state and control variable bounds
    - water quality discretization parameters
    - frictional loss formula
    - optimization objective

"""
function make_prob_data(network::Network, Δt, Δk, sim_days, disc_method; pmin::Int64=15, pmax::Int64=200, Qmax_mul=1000, Qmin_mul=-1000, x_wq_bounds=(0.5, 3), u_wq_bounds=(0, 5), QA=false, quadratic_approx="relative", J=nothing, kb=0.5, kw=0, x_wq_0=0, obj_type="AZP", max_pump_switch::Int64, c_elec::Union{Matrix{Float64}, Vector{Float64}})

    n_t = Int(get_hydraulic_time_steps(network, sim_days, Δk))
    sim_type = "hydraulic"
    sim_results = epanet_solver(network, sim_type; sim_days=sim_days, Δk=Δk)

    # h bounds at junctions
    j_demand = all(>(0), network.d, dims=2)
    Hmin_j = repeat([j ∈ j_demand ? pmin + network.elev[network.junction_idx[j]] : network.elev[network.junction_idx[j]] for j in 1:network.n_j], 1, n_t)
    Hmax_j = repeat([pmax + network.elev[network.junction_idx[j]] for j in 1:network.n_j], 1, n_t)

    # h bounds at tanks
    Hmin_tk = repeat(network.tank_min .+ network.elev[network.tank_idx], 1, n_t +1)
    Hmax_tk = repeat(network.tank_max .+ network.elev[network.tank_idx], 1, n_t + 1)
    tk_init = network.tank_init .+ network.elev[network.tank_idx]

    # q bounds across links
    v_init = Matrix(getfield(sim_results, :velocity))'
    v_max = vmax(network, v_init[2:end, :])
    link_area = (pi .* network.D .^ 2) ./ 4
    Qmin = -1 * v_max .* link_area .* 1000
    Qmax = v_max .* link_area .* 1000

    # q bounds for pumps
    if !isempty(network.pump_idx)
        for (i, p) ∈ enumerate(network.pump_idx)
            p = Polynomial([network.pump_C[i], network.pump_B[i], network.pump_A[i]])
            r = roots(p)
            r = r[r .> 0][1]
            Qmax[network.pump_idx, :] .= r .* 1000
            Qmin[network.pump_idx, :] .= 0
        end
    end

    # u bounds at booster locations
    b_loc, _ = get_booster_inputs(network, network.name, sim_days, Δk, Δt) # booster control locations
    Umin_b = u_wq_bounds[1] * ones(length(b_loc))
    Umax_b = u_wq_bounds[2] * ones(length(b_loc))

    # pump locations
    m_loc = findall(x -> x==1, network.B_pump)

    # water quality bounds at nodes
    Xmin_r = 0 * ones(network.n_r)
    Xmax_r = Inf * ones(network.n_r)
    Xmin_j = x_wq_bounds[1] * ones(network.n_j)
    Xmax_j = x_wq_bounds[2] * ones(network.n_j)
    Xmin_tk = x_wq_bounds[1] * ones(network.n_tk)
    Xmax_tk = x_wq_bounds[2] * ones(network.n_tk)

    # frictional loss model
    if QA
        a, b = quadratic_app(network, v_max, quadratic_approx)
        θmin = (a .* abs.(Qmin) .^ 2 .+ b .* abs.(Qmin)) .* -1
        θmax = a .* Qmax .^ 2 .+ b .* Qmax
    else
        a, b = nothing, nothing
        θmin = (network.r .* Qmax .* abs.(Qmax) .^ (network.nexp .- 1)) .* -1
        θmax = network.r .* Qmax .* abs.(Qmax) .^ (network.nexp .- 1)
    end

    θmin[network.pump_idx, :] .= network.pump_C .* -1.5
    θmax[network.pump_idx, :] .= network.pump_C .* 1.5

    # set discretization parameters and variables
    s_p = []
    n_s = []
    vel_p = v_init[network.pipe_idx, :]
    L_p = network.L[network.pipe_idx]
    if J === nothing
        vel_p_max = maximum(vel_p, dims=2)
        s_p = vec(L_p ./ (vel_p_max .* Δt))
        s_p = floor.(s_p)
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
    k_t = zeros(T)
    for t ∈ 1:T
        k_t[t] = searchsortedfirst(k_set, t*Δt) - 1
    end
    k_t = Int.(k_t)


    return OptParams(
        obj_type=obj_type,
        disc_method=disc_method,
        Qmin=Qmin,
        Qmax=Qmax,
        pmin=pmin,
        Hmin_j=Hmin_j,
        Hmax_j=Hmax_j,
        Hmin_tk=Hmin_tk,
        Hmax_tk=Hmax_tk,
        tk_init=tk_init,
        Xmin_r=Xmin_r,
        Xmax_r=Xmax_r,
        Xmin_j=Xmin_j,
        Xmax_j=Xmax_j,
        Xmin_tk=Xmin_tk,
        Xmax_tk=Xmax_tk,
        Umin_b=Umin_b,
        Umax_b=Umax_b,
        b_loc=b_loc,
        m_loc=m_loc,
        a=a,
        b=b,
        Δx_p=Δx_p,
        s_p=s_p, 
        sim_days=sim_days,
        T=T,
        k_t=k_t,
        Δt=Δt,
        Δk=Δk,
        kb=kb,
        QA=QA,
        θmin=θmin,
        θmax=θmax,
        x_wq_0=x_wq_0,
        max_pump_switch=max_pump_switch,
        c_elec=c_elec
    )
end



"""
Determine the maximum v for a network
    vmul = multipliers to use
"""
function vmax(network, v_init; vmul::Vector{<:Real}=[1.5, 2, 3, 4, 5, 6])
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

    a = zeros(network.n_l, 1)
    b = zeros(network.n_l, 1)
    link_area = (pi .* network.D .^ 2) ./ 4
    Q = v_max .* link_area .* 1000 # convert to L/s

    if network.simtype == "H-W"

        r = network.r
        n = network.nexp[1]

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

        a[network.valve_idx, :] .= network.r[network.valve_idx]
        b[network.valve_idx, :] .= 0
        a[network.pump_idx, :] .= 1e-4
        b[network.pump_idx, :] .= 0

        return a, b

    end

end




"""
Function for getting starting point values for optimal control problem
"""
function get_starting_point(network, opt_params)

    sim_type = "hydraulic"
    sim_results = epanet_solver(network, sim_type; sim_days=opt_params.sim_days, Δk=opt_params.Δk)
    h = Matrix(sim_results.head[:, 2:end])'
    q = Matrix(sim_results.flow[:, 2:end])'

    # initial flow values
    q_0 = q
    q⁺_0 = abs.(q_0); q⁺_0[q_0 .<= 0] .= 0
    q⁻_0 = abs.(q_0); q⁻_0[q_0 .>= 0] .= 0
    @assert(q_0 == q⁺_0 - q⁻_0)
    z_0 = zeros(size(q_0)); z_0[q_0 .< 0] .= 0; z_0[q_0 .> 0] .= 1

    # initial pump slack variables
    u_m_0 = zeros(network.n_m, size(q_0, 2))
    A = hcat(network.A12, network.A10_res, network.A10_tank)
    for i ∈ network.pump_idx
        node_start = findfirst(x -> x == -1, A[i, :])
        node_end = findfirst(x -> x == 1, A[i, :])
        u_m_0[findfirst(x -> x == i, network.pump_idx), :] = h[node_end, :] - h[node_start, :]
    end

    # initial head loss/gain values
    a, b, = opt_params.a, opt_params.b
    r, nexp = network.r, network.nexp
    θ⁺_0 = zeros(size(q_0))
    θ⁻_0 = zeros(size(q_0))
    for k ∈ 1:size(q, 2)
        # head loss across pipe and valve links
        for i ∈ vcat(network.pipe_idx, network.valve_idx)
            if opt_params.QA
                θ⁺_0[i, k] == a[i, k] * q⁺_0[i, k]^2 + b[i, k] * q⁺_0[i, k]
                θ⁻_0[i, k] == a[i, k] * q⁻_0[i, k]^2 + b[i, k] * q⁻_0[i, k]
            else
                θ⁺_0[i, k] == r[i] * q⁺_0[i, k]^nexp[i]
                θ⁻_0[i, k] == r[i] * q⁻_0[i, k]^nexp[i]
            end
        end
        # head gain across pump links
        for i ∈ network.pump_idx
            θ⁺_0[i, k] == -1 .* (network.pump_A[findfirst(x -> x == i, network.pump_idx)] * (q⁺_0[i, k] / 1000)^2 + network.pump_B[findfirst(x -> x == i, network.pump_idx)] * (q⁺_0[i, k] / 1000) + network.pump_C[findfirst(x -> x == i, network.pump_idx)]) + u_m_0[findfirst(x -> x == i, network.pump_idx), k]
        end
    end
    θ_0 = θ⁺_0 - θ⁻_0
        

    # initial head values
    h_j_0 = h[network.junction_idx, :]
    h_tk_0 = opt_params.tk_init


    return h_j_0, h_tk_0, q_0, q⁺_0, q⁻_0, z_0, θ_0, θ⁺_0, θ⁻_0, u_m_0
end







"""
Main function for solving the joint hydraulic and water quality optimization problem
"""
function optimize_hydraulic_wq(network::Network, opt_params::OptParams, sim_days, Δt, Δk, source_cl, b_loc, x0; x_wq_0=0.5, solver="Gurobi", integer=true, warm_start=false, heuristic=false, optimize_wq=false, kb=0.5, kw=0.1, disc_method="explicit-central", x_bounds=(0.5, 3), u_bounds=(0, 5))

    ##### SET OPTIMIZATION PARAMETERS #####
    # unload network data
    net_name = network.name
    n_r = network.n_r
    n_j = network.n_j
    n_tk = network.n_tk
    n_n = network.n_n
    n_m = network.n_m
    n_v = network.n_v
    n_p = network.n_p
    n_l = network.n_l
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
    r = network.r
    nexp = network.nexp
    h0 = network.h0
    d = network.d
    A12 = network.A12
    A10_res = network.A10_res
    A10_tank = network.A10_tank
    A_inc = hcat(A12, A10_res, A10_tank)
    tk_elev = network.elev[tank_idx]
    tk_area = network.tank_area
    pump_A = network.pump_A
    pump_B = network.pump_B
    pump_C = network.pump_C
    # core_links = network.core_links
    pump_η = 0.78 # fixed pump efficiency

    # unload optimization parameters
    disc_method = opt_params.disc_method
    Qmin = opt_params.Qmin
    Qmax = opt_params.Qmax
    Hmin_j = opt_params.Hmin_j
    Hmax_j = opt_params.Hmax_j
    Hmin_tk = opt_params.Hmin_tk
    Hmax_tk = opt_params.Hmax_tk
    tk_init = opt_params.tk_init
    Xmin_r = opt_params.Xmin_r
    Xmax_r = opt_params.Xmax_r
    Xmin_j = opt_params.Xmin_j
    Xmax_j = opt_params.Xmax_j
    Xmin_tk = opt_params.Xmin_tk
    Xmax_tk = opt_params.Xmax_tk
    Umin_b = opt_params.Umin_b
    Umax_b = opt_params.Umax_b
    b_loc = opt_params.b_loc
    m_loc = opt_params.m_loc
    QA = opt_params.QA
    θmin = opt_params.θmin
    θmax = opt_params.θmax
    a = opt_params.a
    b = opt_params.b

    obj_type = opt_params.obj_type
    max_pump_switch = opt_params.max_pump_switch

    # simulation times
    T = opt_params.T
    k_t = opt_params.k_t
    n_t = network.n_t
    #print("Imported k_t is of size...",size(k_t))
    #n_t = length(k_t)
    if optimize_wq
        T_k = Int(T*3600 / Δt) # number of discrete time steps
        k_t = zeros(1, T_k+1)
        k_set = (0:n_t-1) .* 3600
        for t ∈ 1:T_k
            k_t[t] = searchsortedfirst(k_set, t*Δt) - 1
        end
        k_t[end] = k_t[1]
        k_t = Int.(k_t)
    end

    if optimize_wq
        # assign constant parameters (copied from optimize_wq_fix_hyd)
        kb = (kb/3600/24) # units = 1/second
        kw = (kw/3600/24) # units = 1/second
        ν = 1.0533e-6 # kinematic velocity of water in m^2/s
        ϵ_reg = 1e-3 # small regularization value to avoid division by zero

        # define pipe segments
        vel_p_max = 4*(maximum([abs.(Qmin[pipe_idx,:]) abs.(Qmax[pipe_idx,:])],dims=2)/1000)./(π*network.D[pipe_idx,:].^2)
        #print("Size of vel_p_max is...",size(vel_p_max))
        #print("Size of L_p is...",size(L_p))
        s_p = L_p ./ (vel_p_max .* Δt)
        s_p = floor.(Int, s_p)
        s_p[s_p .== 0] .= 1
        n_s = sum(s_p)
        Δx_p = L_p ./ s_p
    end

    # electricity costs
    c_elec = opt_params.c_elec 

    # modify operational data times
    if size(h0, 2) != n_t
        print("h0 is...",h0)
        print("n_t is...",n_t)
        mult = n_t / size(h0, 2)
        if mult > 1
            h0 = repeat(h0, 1, Int(mult))
            d = repeat(d, 1, Int(mult))
        else
            @error "Operational data must be at least as long as the simulation time."
        end
    end

    # find junction, reservoir, and tank nodes
    junction_map = Dict(i => findall(A12[i, :].!=0) for i in 1:n_l)
    reservoir_map = Dict(i => findall(A10_res[i, :].!=0) for i in 1:n_l)
    tank_map = Dict(i => findall(A10_tank[i, :].!=0) for i in 1:n_l)
    link_junction_map = Dict(i => findall(A12'[i, :].!=0) for i in 1:n_j)
    link_tank_map = Dict(i => findall(A10_tank'[i, :].!=0) for i in 1:n_tk)





    ##### BUILD & SOLVE OPTIMIZATION MODEL #####

    if solver == "Gurobi"
        
        ### GUROBI
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "TimeLimit", 60) # 21600) # 6-hour time limit
        # set_optimizer_attribute(model,"Method", 2)
        # set_optimizer_attribute(model,"Presolve", 0)
        # set_optimizer_attribute(model,"Crossover", 0)
        # set_optimizer_attribute(model,"NumericFocus", 3)
        # set_optimizer_attribute(model,"NonConvex", 2)
        set_optimizer_attribute(model, "FeasibilityTol", 1e-4)
        set_optimizer_attribute(model, "IntFeasTol", 1e-4)
        set_optimizer_attribute(model, "OptimalityTol", 1e-4)
        # set_optimizer_attribute(model, "MIPFocus", 1)
        set_optimizer_attribute(model, "MIPGap", 0.05)
        # set_optimizer_attribute(model, "Threads", 4)
        if heuristic
            set_optimizer_attribute(model, "NoRelHeurTime", 60*15)
        end
        
        # set_silent(model)

    elseif solver == "Ipopt"

        ### IPOPT
        model = Model(Ipopt.Optimizer)
        # set_optimizer_attribute(model, "max_iter", 5000)
        set_optimizer_attribute(model, "warm_start_init_point", "yes")
        set_optimizer_attribute(model, "linear_solver", "ma57")
        # set_optimizer_attribute(model, "linear_solver", "spral")
        # set_optimizer_attribute(model, "linear_solver", "ma97")
        # set_attribute(model, "linear_solver", "pardiso")
        set_optimizer_attribute(model, "mu_strategy", "adaptive")
        set_optimizer_attribute(model, "mu_oracle", "quality-function")
        set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
        # set_optimizer_attribute(model, "tol", 1e-4)
        # set_optimizer_attribute(model, "constr_viol_tol", 1e-4)
        # set_optimizer_attribute(model, "fast_step_computation", "yes")
        set_optimizer_attribute(model, "print_level", 5)

        # @error "Optimization solver $solver has not been implemented in this work."

        # opt_results = ipopt_solver(network, opt_params, x_wq_0)


    elseif solver == "SCIP"

        ### SCIP
        model = Model(SCIP.Optimizer)
        set_optimizer_attribute(model, "limits/time", 3600.0) # 1-hour time limit
        set_attribute(model, "limits/gap", 0.05)

    else

        @error "Optimization solver $solver has not been implemented in this work."
        
    end



    ### define variables

    # hydraulic variables
    @variable(model, Hmin_j[i, k] ≤ h_j[i=1:n_j, k=1:n_t] ≤ Hmax_j[i, k])
    @variable(model, Hmin_tk[i, k] ≤ h_tk[i=1:n_tk, k=1:n_t+1] ≤ Hmax_tk[i, k])
    #@variable(model, q[i=1:n_l, k=1:n_t])
    @variable(model, Qmin[i, k] ≤ q[i=1:n_l, k=1:n_t] ≤ Qmax[i, k])
    @variable(model, 0 ≤ q⁺[i=1:n_l, k=1:n_t])
    @variable(model, 0 ≤ q⁻[i=1:n_l, k=1:n_t])
    #@variable(model, q⁺[i=1:n_l, k=1:n_t])
    #@variable(model, q⁻[i=1:n_l, k=1:n_t])
    # @variable(model, 0 ≤ s[i=1:n_l, k=1:n_t]) # something to do with water quality
    @variable(model, θ[i=1:n_l, k=1:n_t])
    @variable(model, θ⁺[i=1:n_l, k=1:n_t])  
    @variable(model, θ⁻[i=1:n_l, k=1:n_t])
    # @variable(model, u_m[i=1:n_m, k=1:n_t]) # n_m is the number of pumps

    if solver ∈ ["Gurobi", "SCIP"]  && integer
        @variable(model, z[i=1:n_l, k=1:n_t], binary=true) # integer variables for pipe flow directions?
        # for i ∈ pump_idx
        #     fix.(z[i], 1; force=true)
        # end
    elseif solver ∈ ["Gurobi", "SCIP"] && !integer
        @variable(model, 0 ≤ z[i=1:n_l, k=1:n_t] ≤ 1)
        # @constraint(model, complementarity[i=1:n_l, k=1:n_t], z[i, k] * (1 - z[i, k]) == 0.0)
    else
        @variable(model, 0 ≤ z[i=1:n_l, k=1:n_t] ≤ 1)
        # @constraint(model, flow_complementarity[i=1:n_l, k=1:n_t], z[i, k] * (1 - z[i, k]) ≤ 1e-4)
    end

    if !isempty(pump_idx)
        @variable(model, 0 ≤ pump_switch[i=1:n_m, k=1:n_t-1] ≤ 1)
    end


    # water quality variables (copied and pasted from optimize_wq_fix_hyd)
    if optimize_wq
        # tank level variables
        @variable(model, 0 ≤ V_tk[i=1:n_tk, k=1:n_t+1])
        # actual wq variables (chlorine concentrations)
        @variable(model, x_bounds[1] ≤ c_r[i=1:n_r, t=1:T_k+1] ≤ x_bounds[2])
        @variable(model, x_bounds[1] ≤ c_j[i=1:n_j, t=1:T_k+1] ≤ x_bounds[2])
        @variable(model, x_bounds[1] ≤ c_tk[i=1:n_tk, t=1:T_k+1] ≤ x_bounds[2])
        @variable(model, x_bounds[1] ≤ c_m[i=1:n_m, t=1:T_k+1] ≤ x_bounds[2])
        @variable(model, x_bounds[1] ≤ c_v[i=1:n_v, t=1:T_k+1] ≤ x_bounds[2])
        @variable(model, x_bounds[1] ≤ c_p[i=1:n_s, t=1:T_k+1] ≤ x_bounds[2])
        @variable(model, u_bounds[1] ≤ u[i=1:n_j, t=1:n_t] ≤ u_bounds[2])
        # additional hydraulic variables (now that hydraulic conditions are NOT fixed...)
        # @variable(model, -1 ≤ qdir[i=1:n_l, k=1:n_t] ≤ 1) # qdir = z*2 - 1
        @variable(model, 0 ≤ λ⁺_p[i=1:n_p, k=1:n_t]) #
        @variable(model, 0 ≤ λ⁻_p[i=1:n_p, k=1:n_t]) # λ_p: courant number, positive, defined as a linear function of q⁺ and q⁻ in pipe_idx
        # @variable(model, 0 ≤ c_in[i=1:n_j, j=1:n_l, k=1:n_t])
        # @variable(model, 0 ≤ c_up[i=1:n_s, t=2:T_k+1])
        # @variable(model, 0 ≤ q_tk_out[i=1:n_tk, k=1:n_t])
    end


    ### define constraints

    # initial conditions
    for i ∈ tank_idx
        fix.(h_tk[findfirst(x -> x == i, tank_idx), 1], tk_init[findfirst(x -> x == i, tank_idx)]; force=true)
    end

    # energy conservation
    if !isempty(tank_idx)
        @constraint(model, energy_conservation[i=1:n_l, k=1:n_t], θ[i, k] + sum(A12[i, j]*h_j[j, k] for j ∈ junction_map[i]) + sum(A10_res[i, j]*h0[j, k] for j ∈ reservoir_map[i]) + sum(A10_tank[i, j]*h_tk[j, k] for j ∈ tank_map[i]) == 0.0)
    else
        @constraint(model, energy_conservation[i=1:n_l, k=1:n_t], θ[i, k] + sum(A12[i, j]*h_j[j, k] for j ∈ junction_map[i]) + sum(A10_res[i, j]*h0[j, k] for j ∈ reservoir_map[i]) == 0.0)
    end

    # head loss/gain model constraints
    for k ∈ 1:n_t
        # head loss across pipe and valve links
        for i ∈ vcat(pipe_idx, valve_idx)
            if QA
                @constraint(model, θ⁺[i, k] == a[i, k] * q⁺[i, k]^2 + b[i, k] * q⁺[i, k])
                @constraint(model, θ⁻[i, k] == a[i, k] * q⁻[i, k]^2 + b[i, k] * q⁻[i, k])
            else
                @constraint(model, θ⁺[i, k] == r[i] * q⁺[i, k]^nexp[i])
                @constraint(model, θ⁻[i, k] == r[i] * q⁻[i, k]^nexp[i])
            end
        end
        # head gain across pump links
        for i ∈ pump_idx
            @constraint(model, θ⁺[i, k] == -1 .* (pump_A[findfirst(x -> x == i, pump_idx)] * (q⁺[i, k] / 1000)^2 + pump_B[findfirst(x -> x == i, pump_idx)] * (q⁺[i, k] / 1000) + pump_C[findfirst(x -> x == i, pump_idx)]) ) #+ u_m[findfirst(x -> x == i, pump_idx), k])
        end
    end
        

    # flow and head loss direction constraints
    @constraint(model, flow_value[i=1:n_l, k=1:n_t], q⁺[i, k] - q⁻[i, k] == q[i, k])
    # @constraint(model, flow_value_abs[i=1:n_l, k=1:n_t], q⁺[i, k] + q⁻[i, k] == s[i, k])
    @constraint(model, head_loss_value[i=1:n_l, k=1:n_t], θ⁺[i, k] - θ⁻[i, k] == θ[i, k])


    # complementarity constraints for flow direction: stopped here!
    if solver ∈ ["Gurobi", "SCIP"]
        # @constraint(model, flow_direction_pos[i=core_links, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        # @constraint(model, flow_direction_neg[i=core_links, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
        @constraint(model, flow_direction_pos[i=1:n_l, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        @constraint(model, flow_direction_neg[i=1:n_l, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
        # @constraint(model, flow_direction[i=1:n_l, k=1:n_t], [q⁻[i, k], q⁺[i, k]] in SOS1())
        # @constraint(model, flow_direction[i=core_links, k=1:n_t], [q⁻[i, k], q⁺[i, k]] in SOS1())
        # @constraint(model, head_loss_direction_pos[i=1:n_l, k=1:n_t], θ⁺[i, k] ≤ z[i, k] * θmax[i, k])
        # @constraint(model, head_loss_direction_neg[i=1:n_l, k=1:n_t], θ⁻[i, k] ≤ (1 - z[i, k]) * abs(θmin[i, k]))
        # @constraint(model, head_loss_direction[i=1:n_l, k=1:n_t], [ θ⁻[i, k], θ⁺[i, k]] in SOS1())
    elseif solver == "Ipopt"
        @constraint(model, flow_direction[i=1:n_l, k=1:n_t], q⁺[i, k] * q⁻[i, k] ≤ 0)
        # @constraint(model, flow_direction[i=core_links, k=1:n_t], q⁺[i, k] * q⁻[i, k] ≤ 0)
        # @constraint(model, flow_direction_pos[i=1:n_l, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        # @constraint(model, flow_direction_neg[i=1:n_l, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
    end

    # complementarity constraints for pump status and maximum number of pump switches
    if !isempty(pump_idx)
        if solver == "Gurobi"
            # @constraint(model, pump_status[i=pump_idx, k=1:n_t], [u_m[findfirst(x -> x == i, pump_idx), k], q⁺[i, k]] in SOS1())
            # @constraint(model, pump_status[i=1:n_l, k=1:n_t],  u_m[i, k] * q⁺[i, k] ≤ 1e-6)
            @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  θ⁻[i, k] ≤ 100*(1 - z[i, k]) )
            @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  θ⁻[i, k] ≥ -100*(1 - z[i, k]) )
            # @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≤ 1e-6)
            # @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≥ -1e-6)
            
            # maximum number of pump switches
            @constraint(model, max_pump_switch[i=1:n_m], sum(pump_switch[i, k] for k in 1:n_t-1) <= max_pump_switch)
            @constraint(model, max_pump_switch_ub[i=1:n_m, k=1:n_t-1], pump_switch[i, k] >= z[pump_idx[i], k+1] - z[pump_idx[i], k])  
            @constraint(model, max_pump_switch_lb[i=1:n_m, k=1:n_t-1], pump_switch[i, k] >= -(z[pump_idx[i], k+1] - z[pump_idx[i], k]))  # Lower bound for the absolute value

            #= fix pump status/schedule for testing
            @constraint(model, pump_schedule1[i=1:n_m, k=1:10], z[pump_idx[i], k] == 1)
            @constraint(model, pump_schedule2[i=1:n_m, k=11:16], z[pump_idx[i], k] == 0)
            @constraint(model, pump_schedule3[i=1:n_m, k=17:n_t], z[pump_idx[i], k] == 1) =#
        elseif solver ∈ ["Ipopt", "SCIP"]
            @constraint(model, pump_status[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] == 0)
            # @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≤ 1e-6)
            # @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≥ -1e-6)
        end
    end

    # tank balance
    if !isempty(tank_idx)
        @constraint(model, tank_balance[i=1:n_tk, k=1:n_t], h_tk[i, k+1] == h_tk[i, k] + sum(A10_tank[j, i]*(q[j, k] / 1000) for j ∈ link_tank_map[i]) * (Δk / tk_area[i]))
    end

    # mass conservation
    @constraint(model, mass_conservation[i=1:n_j, k=1:n_t], sum(A12'[i, j]*q[j, k] for j ∈ link_junction_map[i]) == d[i, k])

    #= flow bounds
    @constraint(model, q_bounds_up[i=1:n_l , k=1:n_t],  q[i, k] ≤ Qmax[i, k] )
    @constraint(model, q_bounds_lo[i=1:n_l , k=1:n_t],  Qmin[i, k] ≤ q[i, k] )
    @constraint(model, q_pos_bounds[i=1:n_l , k=1:n_t], 0 ≤ q⁺[i, k] )
    @constraint(model, q_neg_bounds[i=1:n_l , k=1:n_t], 0 ≤ q⁻[i, k] ) =#


    # water quality constraints (copied and pasted from optimize_wq_fix_hyd)
    if optimize_wq
        # define constraints on tank volume
        @constraint(model, tank_volume[i=1:n_tk, k=1:n_t], V_tk[i,k] == h_tk[i,k] - tk_elev[i]*tk_area[i]*1000)

        # define constraints on basic hydraulic variables and new hydraulic variables (required for wq optimization)
        # @constraint(model, qdir .== 2*z - 1)
        #@constraint(model, λ_p_def, λ_p .== (4*((q⁺[pipe_idx,:]+q⁻[pipe_idx,:])/1000)./(π*network.D.^2)) ./ repeat(Δx_p, 1, n_t) .* Δt)
        #print("The size of vel_p is...", size((4*(q⁺[pipe_idx,:]/1000)./(π*network.D[pipe_idx,:].^2))))
        #print("The size of the denominator is...", size(repeat(Δx_p, 1, n_t)))
        @constraint(model, λ⁺_p_def, λ⁺_p .== (4*(q⁺[pipe_idx,:]/1000)./(π*network.D[pipe_idx,:].^2)) ./ repeat(Δx_p, 1, n_t) .* Δt)
        @constraint(model, λ⁻_p_def, λ⁻_p .== (4*(q⁻[pipe_idx,:]/1000)./(π*network.D[pipe_idx,:].^2)) ./ repeat(Δx_p, 1, n_t) .* Δt)
        #λ_p = λ_p .* qdir[pipe_idx, :]
        # here, we consider advection-dominant PDE discretization schemes: ignore Re
        #@constraint(model, Re .== (4 .* (q[pipe_idx, :] ./ 1000)) ./ (π .* D[pipe_idx, :] .* ν))

        
        # define new variables c_in and q_out for the wq mass balance based on z
        #= @constraint(model, def_c_in[i=1:n_j, j=1:n_l, t=2:T_k+1], 
            c_in[i,j,t] .== (
                j in pipe_idx ? 
                    c_p[sum(s_p[1:findfirst(x -> x == j, pipe_idx)]),t]*z[j,k_t(t-1)] +
                    c_p[sum(s_p[1:findfirst(x -> x == j, pipe_idx) - 1]) + 1,t]*(1-z[j,k_t(t-1)]) :
                j in pump_idx ? 
                    c_m[findfirst(x -> x == j, pump_idx), t] :
                    c_v[findfirst(x -> x == j, valve_idx), t]
            )
        ) =#
        
        #= @constraint(model, def_q_tk_out[i=1:n_tk, k=1:n_t], 
            q_tk_out[i,k] == sum(q⁻[findall(x -> x == 1, A_inc[:, tank_idx[i]]), k]) 
            + sum(q⁺[findall(x -> x == -1, A_inc[:, tank_idx[i]]), k])
        ) =#


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
        @constraint(model, wq_junction_balance[i=1:n_j, t=2:T_k+1],
            ( c_j[i, t] - u[i, k_t[t-1]] )*(
                d[i, k_t[t-1]] +
                sum(q⁻[j, k_t[t-1]] for j in findall(x -> x == 1, A_inc[:, junction_idx[i]])) +
                sum(q⁺[j, k_t[t-1]] for j in findall(x -> x == -1, A_inc[:, junction_idx[i]])) +
                ϵ_reg 
            ) == 
            # sum( (q⁺[:, k_t[t-1]] + q⁻[:, k_t[t-1]] + ϵ_reg) .* c_in[i,:,t] )
            sum(
                (q⁺[j, k_t[t-1]] + ϵ_reg) * (
                    j in pipe_idx ? 
                        c_p[ sum(s_p[1:findfirst(x -> x == j, pipe_idx)]), t ] :
                    j in pump_idx ? 
                        c_m[findfirst(x -> x == j, pump_idx), t] :
                        c_v[findfirst(x -> x == j, valve_idx), t]
                ) for j in findall(x -> x == 1, A_inc[:, junction_idx[i]])
            ) +
            sum(
                (q⁻[j, k_t[t-1]] + ϵ_reg) * (
                    j in pipe_idx ? 
                        c_p[ sum(s_p[1:findfirst(x -> x == j, pipe_idx) - 1]) + 1, t ] :
                    j in pump_idx ? 
                        c_m[findfirst(x -> x == j, pump_idx), t] :
                        c_v[findfirst(x -> x == j, valve_idx), t]
                ) for j in findall(x -> x == -1, A_inc[:, junction_idx[i]])
            )
        )

        # tank mass balance
        print("n_tk is...", n_tk)
        print("T_k+1 is...", T_k+1)
        print("Size of c_tk is...",size(c_tk))
        print("Size of V_tk is...",size(V_tk))
        print("Size of k_t is...",size(k_t))
        @constraint(model, wq_tank_balance[i=1:n_tk, t=2:T_k+1],
            c_tk[i, t]*V_tk[i, k_t[t]] == ( # previous time step
                c_tk[i, t-1] * V_tk[i, k_t[t-1]] -
                ( # outflow
                    c_tk[i, t-1] * Δt * ( sum(q⁻[findall(x -> x == 1, A_inc[:, tank_idx[i]])]) 
                    + sum(q⁺[findall(x -> x == -1, A_inc[:, tank_idx[i]])]) )
                ) +
                ( # inflow
                    sum(
                        q⁺[j, k_t[t-1]] * Δt * (
                            j in pipe_idx ? 
                                c_p[
                                    sum(s_p[1:findfirst(x -> x == j, pipe_idx)]), t-1
                                ] :
                            (j in pump_idx ? 
                                c_m[findfirst(x -> x == j, pump_idx), t-1] :
                                c_v[findfirst(x -> x == j, valve_idx), t-1]
                            )
                        ) for j in findall(x -> x == 1, A_inc[:, tank_idx[i]])
                    ) +
                    sum(
                        q⁻[j, k_t[t-1]] * Δt * (
                            j in pipe_idx ? 
                                c_p[
                                    sum(s_p[1:findfirst(x -> x == j, pipe_idx) - 1]) + 1, t-1
                                ] :
                            (j in pump_idx ? 
                                c_m[findfirst(x -> x == j, pump_idx), t-1] :
                                c_v[findfirst(x -> x == j, valve_idx), t-1]
                            )
                        ) for j in findall(x -> x == -1, A_inc[:, tank_idx[i]])
                    )
                ) +
                ( # decay
                    -1 * c_tk[i, t-1] * kb * V_tk[i, k_t[t-1]] * Δt
                )
            )
        )

        #= pump mass balance:
        what happens if flow reverses up to pump downstream node? 
        this is not currently handled, but should not occur unless the pump downstream node has a positive demand =#
        @constraint(model, wq_pump_balance[i=1:n_m, t=2:T_k+1],
            c_m[i, t] == begin
                node_idx = findall(x -> x == -1, A_inc[pump_idx[i], :])[1]
                c_up = node_idx ∈ reservoir_idx ? 
                    c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                    node_idx ∈ junction_idx ? 
                        c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t]
                c_up # *z[pump_idx[i],k_t[t-1]] 
            end # + term for when flow reverses all the way back up to pump downstream node?
        )

        #= valve mass balance:
        are we talking about non-return valves? can this be ignored for networks without valves, like Net1? =#
        @constraint(model, wq_valve_balance[i=1:n_v, t=2:T_k+1],
            c_v[i, t] == begin
                node_idx = findall(x -> x == -1, A_inc[valve_idx[i], :, k_t[t-1]])[1]
                c_up = node_idx ∈ reservoir_idx ? 
                    c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                    node_idx ∈ junction_idx ? 
                        c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t]
                c_up # * (1-z[findall(x -> x == -1, A_inc[valve_idx[i], :]]) + * z[findall(x -> x == 1, A_inc[valve_idx[i], :]]
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
            # advection-dominant PDE discretization scheme: dispersion is ignored
            #= @constraint(model, c_up_def[i=1:n_s, t=2:T_k+1], 
                c_up[i,t] = i ∈ s_p_start && i ∉ s_p_end ? begin
                        idx = findfirst(x -> x == i, s_p_start) # idx of the pipe the segment i starts off, not the same as p?
                        node_idx = findall(x -> x == -1, A_inc[pipe_idx[idx], :])[1] # find the index of the node at the start of pipe idx
                        (node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t])*z[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]] + c_p[i+1, t]*(1-z[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]])
                    end : 
                    i ∉ s_p_start && i ∈ s_p_end ? begin
                        idx = findfirst(x -> x == i, s_p_end)
                        node_idx = findall(x -> x == 1, A_inc[pipe_idx[idx], :])[1]
                        (node_idx ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t])*(1-z[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]]) + c_p[i-1, t]*z[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]]
                    end : 
                    i ∈ s_p_start && i ∈ s_p_end ? begin
                        idx_start = findfirst(x -> x == i, s_p_start)
                        idx_end = findfirst(x -> x == i, s_p_end)
                        node_idx_start = findall(x -> x == -1, A_inc[pipe_idx[idx_start], :])[1]
                        node_idx_end = findall(x -> x == 1, A_inc[pipe_idx[idx_end], :])[1]
                        (node_idx_start ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx_start, reservoir_idx), t] :
                        node_idx_start ∈ junction_idx ? c_j[findfirst(x -> x == node_idx_start, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx_start, tank_idx), t])*z[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]] +
                        (node_idx_end ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx_end, reservoir_idx), t] :
                        node_idx_end ∈ junction_idx ? c_j[findfirst(x -> x == node_idx_end, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx_end, tank_idx), t])*(1-z[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]])
                    end : 
                    c_p[i-1,t]*z[findlast(x -> x <= i, s_p_start)[1],k_t[t-1]] + c_p[i+1,t]*(1-z[findlast(x -> x <= i, s_p_start)[1],k_t[t-1]])
            )

            @constraint(model, pipe_transport[i=1:n_s, t=2:T_k+1], 
                (1 + λ⁺_p[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]] + λ⁺_p[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]])*c_p[i, t] .== 
                c_p[i, t-1] + λ_p[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]] * c_up[i, t]
            ) =#

            @constraint(model, pipe_transport[i=1:n_s, t=2:T_k+1],  #c_p[i, t-1] * (1 - k_i * Δt)
                (1 + λ⁺_p[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]] + λ⁻_p[findlast(x -> x <= i, s_p_start)[1], k_t[t-1]])*c_p[i, t] .== begin
                    
                    p = findlast(x -> x <= i, s_p_start)[1]
                    
                    i ∈ s_p_start && i ∉ s_p_end ? begin
                    
                        node_idx_start = findall(x -> x == -1, A_inc[pipe_idx[findfirst(x -> x == i, s_p_start)], :])[1]
                        (node_idx_start ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx_start, reservoir_idx), t] :
                        node_idx_start ∈ junction_idx ? c_j[findfirst(x -> x == node_idx_start, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx_start, tank_idx), t])*λ⁺_p[p, k_t[t-1]] + c_p[i+1, t]*λ⁻_p[p, k_t[t-1]]

                    end : i ∉ s_p_start && i ∈ s_p_end ? begin

                        node_idx_end = findall(x -> x == 1, A_inc[pipe_idx[findfirst(x -> x == i, s_p_end)], :])[1]
                        (node_idx_end ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx, reservoir_idx), t] :
                        node_idx_end ∈ junction_idx ? c_j[findfirst(x -> x == node_idx, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx, tank_idx), t])*λ⁻_p[p, k_t[t-1]] + c_p[i-1, t]*λ⁺_p[p, k_t[t-1]]

                    end : i ∈ s_p_start && i ∈ s_p_end ? begin

                        node_idx_start = findall(x -> x == -1, A_inc[pipe_idx[findfirst(x -> x == i, s_p_start)], :])[1]
                        node_idx_end = findall(x -> x == 1, A_inc[pipe_idx[findfirst(x -> x == i, s_p_end)], :])[1]

                        (node_idx_start ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx_start, reservoir_idx), t] :
                        node_idx_start ∈ junction_idx ? c_j[findfirst(x -> x == node_idx_start, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx_start, tank_idx), t])*λ⁺_p[p, k_t[t-1]] +
                        
                        (node_idx_end ∈ reservoir_idx ? c_r[findfirst(x -> x == node_idx_end, reservoir_idx), t] :
                        node_idx_end ∈ junction_idx ? c_j[findfirst(x -> x == node_idx_end, junction_idx), t] :
                        c_tk[findfirst(x -> x == node_idx_end, tank_idx), t])*λ⁻_p[p, k_t[t-1]] 
                        # : nothing

                    end : c_p[i-1, t]*λ⁺_p[p, k_t[t-1]] + c_p[i+1, t]*λ⁻_p[p, k_t[t-1]]
    
                end
            )
        
        else
            @error "Discretization method has not been implemented yet."
            return
        end
    end


    ### define objective functions
    if obj_type == "AZP"
        # insert code here...
    elseif obj_type == "cost"
        # @objective(model, Min, sum(q[i,k] for i ∈ pump_idx, k ∈ 1:n_t))
        #=cost = 0
        for k ∈ 1:n_t
            for i ∈ pump_idx
                cost = cost - Δk*c_elec[k]*9.81*(q⁺[i, k]/1000)*(pump_A[findfirst(x -> x == i, pump_idx)]*(q⁺[i, k]/1000)^2 + pump_B[findfirst(x -> x == i, pump_idx)]*(q⁺[i, k]/1000) + pump_C[findfirst(x -> x == i, pump_idx)])/pump_η
            end
        end =#
        @objective(model, Min, sum(-Δk/3600*c_elec[k]*9.81*(q⁺[i, k]/1000)*θ⁺[i, k]/pump_η for i ∈ pump_idx, k ∈ 1:n_t)) 
        #= from the Matlab code:
        Dh_pumps = -(ap_quad.*q_pumps.^2 + bp_quad.*q_pumps + cp_quad); % Calculate the pump discharge heads.
        efficiencies = 0.78; 
        cost = dth * tou * (9.81*q_pumps.*Dh_pumps) ./ efficiencies; 
        cost = sum(cost); % Calculate total cost.=#
    else
        @objective(model, Min, 0.0)
    end
    # @objective(model, Min, sum(q[i, k] for i ∈ pump_idx, k ∈ 1:n_t))




    ### warm start optimization problem
    if warm_start
        h_j_0, h_tk_0, q_0, q⁺_0, q⁻_0, z_0, θ_0, θ⁺_0, θ⁻_0, u_m_0 = get_starting_point(network, opt_params)
        set_start_value.(h_j, h_j_0)
        # set_start_value.(q, q_0)
        set_start_value.(q⁺, q⁺_0)
        set_start_value.(q⁻, q⁻_0)
        # set_start_value.(z, z_0)
        # set_start_value.(θ, θ_0)
        # set_start_value.(θ⁺, θ⁺_0)
        # set_start_value.(θ⁻, θ⁻_0)
        # set_start_value.(u_m, u_m_0)
        if !isempty(tank_idx)
            set_start_value.(h_tk, h_tk_0)
        end
    end

    ### solve optimization problem
    optimize!(model)
    solution_summary(model)
    term_status = termination_status(model)
    print(term_status)
    accepted_status = [LOCALLY_SOLVED; ALMOST_LOCALLY_SOLVED; OPTIMAL; ALMOST_OPTIMAL; TIME_LIMIT]
    
    if term_status ∉ accepted_status

        @error "Optimization problem did not converge. Please check the model formulation and solver settings."

        if termination_status(model) ∈ [INFEASIBLE; INFEASIBLE_OR_UNBOUNDED]
            
            println("Model is infeasible. Computing IIS...")
            compute_conflict!(model)

            # print conflicting constraints
            #conflict_constraint_list = ConstraintRef[]
            for (F, S) in list_of_constraint_types(model)
                for con in all_constraints(model, F, S)
                    if MOI.get.(model, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                        #push!(conflict_constraint_list, con)
                        println(con)
                    end
                end
            end
            
            #= print confilcting bounds
            #conflicting_bounds = []
            for var in all_variables(model)
                if MOI.get(model, MOI.VariableConflictStatus(), var) == MOI.IN_CONFLICT
                    #push!(conflicting_bounds, ("LowerBound", var))
                    println(var)
                end
            end =#
            
        end

        return OptResults(
            q=zeros(n_l, n_t),
            h_j=zeros(n_j, n_t),
            h_tk=zeros(n_tk, n_t),
            # u_m=zeros(n_l, n_t),
            q⁺=zeros(n_l, n_t),
            q⁻=zeros(n_l, n_t),
            # s=zeros(n_l, n_t),
            θ=zeros(n_l, n_t),
            θ⁺=zeros(n_l, n_t),
            θ⁻=zeros(n_l, n_t),
            z=zeros(n_l, n_t)
        )
    end

    if term_status == TIME_LIMIT
        @error "Maximum time limit reached. Returning best incumbent."
    end

    ### extract results`
    return OptResults(
        q=value.(q),
        h_j=value.(h_j),
        h_tk=value.(h_tk),
        u_m=zeros(n_m,n_t), #value.(u_m),
        q⁺=value.(q⁺),
        q⁻=value.(q⁻),
        s=zeros(n_l,n_t), #value.(s),
        θ=value.(θ),
        θ⁺=value.(θ⁺),
        θ⁻=value.(θ⁻),
        z=value.(z)
    )
end


"""
Main function for solving the joint hydraulic and water quality optimization problem
"""
#= function optimize_hydraulic_admm(network::Network, opt_params::OptParams)

    #= ADMM is an iterative algorithm (try e.g. 50 iterations) with
    - primal variables x
    - auxiliary (complicating) variables y and z
    - and dual variables lambda_y and lambda_z
    - regularization parameter rho 
    Implementation disregards boundary valves for now. =#

    ######### Retrieve network data ############
    # unload network data and time-step data
    T = opt_params.T
    k_t = opt_params.k_t
    n_t = length(k_t)
    #Δt = opt_params.Δt
    Δk = opt_params.Δk # duration of hydraulic time step in s
    dtm = Δk/60  # duration of a time step in minutes

    net_name = network.name
    n_r = network.n_r
    n_j = network.n_j
    n_tk = network.n_tk
    n_n = network.n_n
    n_m = network.n_m
    n_v = network.n_v
    n_p = network.n_p
    n_l = network.n_l 
    A12 = network.A12
    A10_res = network.A10_res
    A10_tank = network.A10_tank
    A_inc = hcat(A12, A10_res, A10_tank)
    #= tk_elev = network.elev[tank_idx]
    tk_area = network.tank_area
    pump_A = network.pump_A
    pump_B = network.pump_B
    pump_C = network.pump_C =#
    pump_η = 0.78 # fixed pump efficiency
    dtm = 60

    ######### Initialize primal and dual variables #########
    # rename everything and go fishing for q, h, ...
    x_0 = get_starting_point(network, opt_params)
    xk = reshape(x_0,2*n_l+n_n+3*n_tk,n_t); 
    yk = xk[ n_l+n_n+n_l+1:n_l+n_n+n_l+n_tk, : ]; 
    zk = [ yk[:,2:end]  yk[:,end]+Δk./tank_areas.*A13'*xk[1:n_l,end] ];
    lambda_yk = zeros(n_tk,n_t);             
    lambda_zk = zeros(n_tk,n_t);             
    lambdak = [ lambda_yk ; lambda_zk ];      
    maxIter = 500; 
    
    y_hist = yk[:];                            
    z_hist = zk[:];                      
    residuals = [];                           
    norm_cumulative_residuals = [];           
    obj = [];                                 
    residual_ratio = [];                      
    dual_residuals = [];                      
    rho_0 = 10; 
    parallel = 1;
    x_update_time = 0;
    y_update_time = 0;
    lambda_update_time = 0;
    
    #= if solver == "branch-and-bound"
        auxdata.scale_residuals = 1;
    end =#
    
    #= min_res = Inf;
    min_obj = Inf;
    erseghe_eps = Inf; =#
    
    #= for the implementation of varying penalty parameters rho,
    % see "Distributed optimization and statistical learning via the 
    % alternating direction method of multipliers", Boyd et al. =#
    
    
    ################## main ADMM loop ##################
    
    rho = 0;
    rho_hist = [];
    
    for k = 1:maxIter
            
        ######### update primal variables xk_t (in parallel) #########
    
        status = zeros(n_t,1);
        print("Starting iteration ",k,".\n")
        start_x_update = toc;
    
        if parallel
            ### not sure how to implement this part in Julia yet
        else
            
            for t = 1:n_t # time steps, e.g. 1 to 24
                # at this point, make sure network time-dependent matrices in network already reflect Δk and n_t
                network_t, params_t = extract_network_params_timestep( network, opt_params, t, Δk, 1, n_t ); 
                params_t.time_step = t;
        
                #= if ~auxdata.bilin_pen_weight_tAll && t > 1
                    temp_data.bilin_pen = 1;
                    temp_data.bilin_pen_weight_vec = zeros(npumps,1);
                end =#
        
                params_t = make_var_and_cons_bounds_admm(network_t,params_t);         
                params_t = add_matrices_admm_constraints(network_t,params_t);
                params_t.rho = rho;
                #params_t.SuppressOutput = 1;
                #params_t.solver = solver;
                
                # iteratively refine the bilinear (complementarity) constraint
                while (abs(uk[data.FSpumpsIdx[1]])>1e-3)*(qk[data.FSpumpsIdx[1]]>1e-6) || 
                    (abs(uk[data.FSpumpsIdx[2]])>1e-3)*(qk[data.FSpumpsIdx[2]]>1e-6)
      
                    params_t.bilin_eps = data.bilin_eps*1e-1;
                    params_t.rho = rho;
                    xk[:,t], info = solve_primal_update( xk[:,t], yk[:,t], zk[:,t], lambda_yk[:,t], lambda_zk[:,t], network_t, params_t, rho ); 
                    
                    qk = x[1:n_l,:];                            
                    uk = x[n_n+n_l+1:2*n_l+n_n,:]; 
                end
        
                if info.status ~= 0 
                    status[t] = 1;
                    error("Optimization did not converge at time step ",t,".\n")
                end
        
            end 
    
        end
        
        if sum(status) > 0
            print("One time step was infeasible!\n")                
        end
    
        obj = [ obj  nlp_obj_NC(xk[:],auxdata) ];
        
        if k == 1
            rho = rho_0;  
        end
    
        start_y_update = toc;
        x_update_time = x_update_time + start_y_update - start_x_update;
    
    
        ######### update auxiliary variables yk and zk #########
    
        temp_data = auxdata;
        temp_data.ps = 0;
        temp_data = make_var_and_cons_bounds_admm_aux(temp_data);         
        temp_data = add_matrices_admm_aux_constraints(temp_data);
        temp_data.rho = rho; 
    
        [ yk, zk, info ] = solve_admm_aux_update( xk, yk, zk, lambda_yk, lambda_zk, temp_data );
        if info.status ~= 0 
            #print("Pause here...\n")
        end
        y_hist = [ y_hist  yk[:] ];
        z_hist = [ z_hist  zk[:] ];
    
        start_lambda_update = toc;
        y_update_time = y_update_time + start_lambda_update - start_y_update;
    
        auxk = [ yk; zk ];
        check_aux_constraints = temp_data.D_aux_mass*auxk[:]; 
        
        ######### compute residuals #########
    
        residualk = [ (xk[n_l+n_n+n_l+1:n_l+n_n+n_l+n_tk,:] - yk) ; 
                      (xk[n_l+n_n+n_l+1:n_l+n_n+n_l+n_tk,:]+Δk./tank_areas.*A13'*xk[1:n_l,:] - zk) ] ;
        
        I_constraint = [ temp_data.I_constraint_y ]; 

        dual_residualk = rho * I_constraint' * ( scale_residuals .* [ yk - reshape(y_hist[:,end-1], n_tk, n_t) ; 
                         zk - reshape(z_hist[:,end-1], n_tk, n_t) ])
    
        y_residuals = residualk[1:n_tk,:];
        z_residuals = residualk[n_tk+1:2*n_tk,:];
        residuals = [ residuals  norm(residualk[:],2) ];
        
        dual_residuals = [dual_residuals  norm( dual_residualk[:], 2) ];  
    
        residual_ratio = [ residual_ratio  norm( scale_residuals[:].*residualk[:], 2 )/norm( dual_residualk[:], 2 ) ];
    
    
        cumulative_x = zeros(2,n_t); 
        cumulative_residuals = zeros(2,n_t);
        for tank = 1:n_tk 
            cumulative_x[tank,:] = xk[n_l+n_n+n_l+tank,1] + (Δk./tank_areas(tank).*A13[:,tank]'*xk[1:n_l,:])*tril(ones(n_t,n_t))';
            cumulative_residuals[tank,:] = cumulative_x[tank,:] - zk[tank,:];
        end
        norm_cumulative_residuals = [ norm_cumulative_residuals  [ norm(cumulative_residuals(1,:),Inf) ; 
                                                                   norm(cumulative_residuals(2,:),Inf) ] ];
    
    
        ###### update dual variables lambda_yk and lamda_zk ######
        
        if all(norm_cumulative_residuals[:,end] <= min_res)
    
            min_res = max(norm_cumulative_residuals[:,end]);
    
            x_inc = xk;
            y_inc = yk;
            z_inc = zk;
            k_inc = k;
            
        end
    
        lambdak = lambdak + rho .* ( scale_residuals.*residualk ) ; 
        lambda_yk = lambdak[ 1:n_tk, : ];
        lambda_zk = lambdak[ n_tk+1:2*n_tk, : ];
    
        lambda_update_time = lambda_update_time + toc - start_lambda_update;
    
        erseghe_eps = max(norm_cumulative_residuals[:,end]);
        
        #= new condition for termination: this is a heuristic method anyways, so
        terminate when the levels given by y and z are within required
        bounds? =#
        if residuals[end] < 0.15 

            x_inc = xk;
            info.lambda_yk = lambda_yk;
            info.lambda_zk = lambda_zk;
            info.iter = k;
            info.x_CPU = x_update_time;
            info.y_CPU = y_update_time;
            info.lambda_CPU = lambda_update_time;
    
            break
    
        elseif k >= maxIter
    
            info.lambda_yk = lambda_yk;
            info.lambda_zk = lambda_zk;
            info.iter = k;

        end
    
    end

    #=
    ##### SET OPTIMIZATION PARAMETERS #####
    # unload network data
    net_name = network.name
    n_r = network.n_r
    n_j = network.n_j
    n_tk = network.n_tk
    n_n = network.n_n
    n_m = network.n_m
    n_v = network.n_v
    n_p = network.n_p
    n_l = network.n_l
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
    r = network.r
    nexp = network.nexp
    h0 = network.h0
    d = network.d
    A12 = network.A12
    A10_res = network.A10_res
    A10_tank = network.A10_tank
    A_inc = hcat(A12, A10_res, A10_tank)
    tk_elev = network.elev[tank_idx]
    tk_area = network.tank_area
    pump_A = network.pump_A
    pump_B = network.pump_B
    pump_C = network.pump_C
    # core_links = network.core_links
    pump_η = 0.78 # fixed pump efficiency

    # unload optimization parameters
    Qmin = opt_params.Qmin
    Qmax = opt_params.Qmax
    Hmin_j = opt_params.Hmin_j
    Hmax_j = opt_params.Hmax_j
    Hmin_tk = opt_params.Hmin_tk
    Hmax_tk = opt_params.Hmax_tk
    #tk_init = opt_params.tk_init
    QA = opt_params.QA
    θmin = opt_params.θmin
    θmax = opt_params.θmax
    a = opt_params.a
    b = opt_params.b
    Δx_p = opt_params.Δx_p
    s_p = opt_params.s_p
    n_s = sum(s_p)
    obj_type = opt_params.obj_type
    #max_pump_switch = opt_params.max_pump_switch

    # simulation times
    T = opt_params.T
    k_t = opt_params.k_t
    n_t = length(k_t)
    Δt = opt_params.Δt

    # electricity costs
    c_elec = opt_params.c_elec 

    # modify operational data times
    if size(h0, 2) != n_t
        mult = n_t / size(h0, 2)
        if mult > 1
            h0 = repeat(h0, 1, Int(mult))
            d = repeat(d, 1, Int(mult))
        else
            @error "Operational data must be at least as long as the simulation time."
        end
    end

    # find junction, reservoir, and tank nodes
    junction_map = Dict(i => findall(A12[i, :].!=0) for i in 1:n_l)
    reservoir_map = Dict(i => findall(A10_res[i, :].!=0) for i in 1:n_l)
    tank_map = Dict(i => findall(A10_tank[i, :].!=0) for i in 1:n_l)
    link_junction_map = Dict(i => findall(A12'[i, :].!=0) for i in 1:n_j)
    link_tank_map = Dict(i => findall(A10_tank'[i, :].!=0) for i in 1:n_tk)


    ##### BUILD & SOLVE OPTIMIZATION MODEL #####

    ### Define high-level model: IPOPT
    model = Model(Ipopt.Optimizer)
    # set_optimizer_attribute(model, "max_iter", 5000)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")
    set_optimizer_attribute(model, "linear_solver", "ma57")
    # set_optimizer_attribute(model, "linear_solver", "spral")
    # set_optimizer_attribute(model, "linear_solver", "ma97")
    # set_attribute(model, "linear_solver", "pardiso")
    set_optimizer_attribute(model, "mu_strategy", "adaptive")
    set_optimizer_attribute(model, "mu_oracle", "quality-function")
    set_optimizer_attribute(model, "fixed_variable_treatment", "make_parameter")
    # set_optimizer_attribute(model, "tol", 1e-4)
    # set_optimizer_attribute(model, "constr_viol_tol", 1e-4)
    # set_optimizer_attribute(model, "fast_step_computation", "yes")
    set_optimizer_attribute(model, "print_level", 5)

    # @error "Optimization solver $solver has not been implemented in this work."

    # opt_results = ipopt_solver(network, opt_params, x_wq_0)


    ### define variables

    # hydraulic variables
    @variable(model, Hmin_j[i, k] ≤ h_j[i=1:n_j, k=1:n_t] ≤ Hmax_j[i, k])
    @variable(model, Hmin_tk[i, k] ≤ h_tk[i=1:n_tk, k=1:n_t+1] ≤ Hmax_tk[i, k])
    #@variable(model, q[i=1:n_l, k=1:n_t])
    @variable(model, Qmin[i, k] ≤ q[i=1:n_l, k=1:n_t] ≤ Qmax[i, k])
    @variable(model, 0 ≤ q⁺[i=1:n_l, k=1:n_t])
    @variable(model, 0 ≤ q⁻[i=1:n_l, k=1:n_t])
    #@variable(model, q⁺[i=1:n_l, k=1:n_t])
    #@variable(model, q⁻[i=1:n_l, k=1:n_t])
    # @variable(model, 0 ≤ s[i=1:n_l, k=1:n_t]) # something to do with water quality
    @variable(model, θ[i=1:n_l, k=1:n_t])
    @variable(model, θ⁺[i=1:n_l, k=1:n_t])  
    @variable(model, θ⁻[i=1:n_l, k=1:n_t])
    # @variable(model, u_m[i=1:n_m, k=1:n_t]) # n_m is the number of pumps

    ### define constraints

    # initial conditions
    for i ∈ tank_idx
        fix.(h_tk[findfirst(x -> x == i, tank_idx), 1], tk_init[findfirst(x -> x == i, tank_idx)]; force=true)
    end

    # energy conservation
    if !isempty(tank_idx)
        @constraint(model, energy_conservation[i=1:n_l, k=1:n_t], θ[i, k] + sum(A12[i, j]*h_j[j, k] for j ∈ junction_map[i]) + sum(A10_res[i, j]*h0[j, k] for j ∈ reservoir_map[i]) + sum(A10_tank[i, j]*h_tk[j, k] for j ∈ tank_map[i]) == 0.0)
    else
        @constraint(model, energy_conservation[i=1:n_l, k=1:n_t], θ[i, k] + sum(A12[i, j]*h_j[j, k] for j ∈ junction_map[i]) + sum(A10_res[i, j]*h0[j, k] for j ∈ reservoir_map[i]) == 0.0)
    end

    # head loss/gain model constraints
    for k ∈ 1:n_t
        # head loss across pipe and valve links
        for i ∈ vcat(pipe_idx, valve_idx)
            if QA
                @constraint(model, θ⁺[i, k] == a[i, k] * q⁺[i, k]^2 + b[i, k] * q⁺[i, k])
                @constraint(model, θ⁻[i, k] == a[i, k] * q⁻[i, k]^2 + b[i, k] * q⁻[i, k])
            else
                @constraint(model, θ⁺[i, k] == r[i] * q⁺[i, k]^nexp[i])
                @constraint(model, θ⁻[i, k] == r[i] * q⁻[i, k]^nexp[i])
            end
        end
        # head gain across pump links
        for i ∈ pump_idx
            @constraint(model, θ⁺[i, k] == -1 .* (pump_A[findfirst(x -> x == i, pump_idx)] * (q⁺[i, k] / 1000)^2 + pump_B[findfirst(x -> x == i, pump_idx)] * (q⁺[i, k] / 1000) + pump_C[findfirst(x -> x == i, pump_idx)]) ) #+ u_m[findfirst(x -> x == i, pump_idx), k])
        end
    end
        

    # flow and head loss direction constraints
    @constraint(model, flow_value[i=1:n_l, k=1:n_t], q⁺[i, k] - q⁻[i, k] == q[i, k])
    # @constraint(model, flow_value_abs[i=1:n_l, k=1:n_t], q⁺[i, k] + q⁻[i, k] == s[i, k])
    @constraint(model, head_loss_value[i=1:n_l, k=1:n_t], θ⁺[i, k] - θ⁻[i, k] == θ[i, k])


    # complementarity constraints for flow direction: stopped here!
    if solver ∈ ["Gurobi", "SCIP"]
        # @constraint(model, flow_direction_pos[i=core_links, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        # @constraint(model, flow_direction_neg[i=core_links, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
        @constraint(model, flow_direction_pos[i=1:n_l, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        @constraint(model, flow_direction_neg[i=1:n_l, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
        # @constraint(model, flow_direction[i=1:n_l, k=1:n_t], [q⁻[i, k], q⁺[i, k]] in SOS1())
        # @constraint(model, flow_direction[i=core_links, k=1:n_t], [q⁻[i, k], q⁺[i, k]] in SOS1())
        # @constraint(model, head_loss_direction_pos[i=1:n_l, k=1:n_t], θ⁺[i, k] ≤ z[i, k] * θmax[i, k])
        # @constraint(model, head_loss_direction_neg[i=1:n_l, k=1:n_t], θ⁻[i, k] ≤ (1 - z[i, k]) * abs(θmin[i, k]))
        # @constraint(model, head_loss_direction[i=1:n_l, k=1:n_t], [ θ⁻[i, k], θ⁺[i, k]] in SOS1())
    elseif solver == "Ipopt"
        @constraint(model, flow_direction[i=1:n_l, k=1:n_t], q⁺[i, k] * q⁻[i, k] ≤ 0)
        # @constraint(model, flow_direction[i=core_links, k=1:n_t], q⁺[i, k] * q⁻[i, k] ≤ 0)
        # @constraint(model, flow_direction_pos[i=1:n_l, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        # @constraint(model, flow_direction_neg[i=1:n_l, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
    end

    # complementarity constraints for pump status and maximum number of pump switches
    if !isempty(pump_idx)
        if solver == "Gurobi"
            # @constraint(model, pump_status[i=pump_idx, k=1:n_t], [u_m[findfirst(x -> x == i, pump_idx), k], q⁺[i, k]] in SOS1())
            # @constraint(model, pump_status[i=1:n_l, k=1:n_t],  u_m[i, k] * q⁺[i, k] ≤ 1e-6)
            @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  θ⁻[i, k] ≤ 100*(1 - z[i, k]) )
            @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  θ⁻[i, k] ≥ -100*(1 - z[i, k]) )
            # @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≤ 1e-6)
            # @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≥ -1e-6)
            
            # maximum number of pump switches
            @constraint(model, max_pump_switch[i=1:n_m], sum(pump_switch[i, k] for k in 1:n_t-1) <= max_pump_switch)
            @constraint(model, max_pump_switch_ub[i=1:n_m, k=1:n_t-1], pump_switch[i, k] >= z[pump_idx[i], k+1] - z[pump_idx[i], k])  
            @constraint(model, max_pump_switch_lb[i=1:n_m, k=1:n_t-1], pump_switch[i, k] >= -(z[pump_idx[i], k+1] - z[pump_idx[i], k]))  # Lower bound for the absolute value

            #= fix pump status/schedule for testing
            @constraint(model, pump_schedule1[i=1:n_m, k=1:10], z[pump_idx[i], k] == 1)
            @constraint(model, pump_schedule2[i=1:n_m, k=11:16], z[pump_idx[i], k] == 0)
            @constraint(model, pump_schedule3[i=1:n_m, k=17:n_t], z[pump_idx[i], k] == 1) =#
        elseif solver ∈ ["Ipopt", "SCIP"]
            @constraint(model, pump_status[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] == 0)
            # @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≤ 1e-6)
            # @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≥ -1e-6)
        end
    end

    # tank balance
    if !isempty(tank_idx)
        @constraint(model, tank_balance[i=1:n_tk, k=1:n_t], h_tk[i, k+1] == h_tk[i, k] + sum(A10_tank[j, i]*(q[j, k] / 1000) for j ∈ link_tank_map[i]) * (Δk / tk_area[i]))
    end

    # mass conservation
    @constraint(model, mass_conservation[i=1:n_j, k=1:n_t], sum(A12'[i, j]*q[j, k] for j ∈ link_junction_map[i]) == d[i, k])

    #= flow bounds
    @constraint(model, q_bounds_up[i=1:n_l , k=1:n_t],  q[i, k] ≤ Qmax[i, k] )
    @constraint(model, q_bounds_lo[i=1:n_l , k=1:n_t],  Qmin[i, k] ≤ q[i, k] )
    @constraint(model, q_pos_bounds[i=1:n_l , k=1:n_t], 0 ≤ q⁺[i, k] )
    @constraint(model, q_neg_bounds[i=1:n_l , k=1:n_t], 0 ≤ q⁻[i, k] ) =#


    # water quality constraints (copied and pasted from optimize_wq_fix_hyd)
    if optimize_wq
        # define constraints on basic hydraulic variables and new hydraulic variables (required for wq optimization)
        @constraint(model, qdir .== 2*z - 1)
        @constraint(model, vel .== 4*((q⁺+q⁻)/1000)./(π*network.D.^2))
        @constraint(model, Re .== (4 .* (q[pipe_idx, :] ./ 1000)) ./ (π .* D[pipe_idx, :] .* ν))

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
    end


    ### define objective functions
    if obj_type == "AZP"
        # insert code here...
    elseif obj_type == "cost"
        # @objective(model, Min, sum(q[i,k] for i ∈ pump_idx, k ∈ 1:n_t))
        #=cost = 0
        for k ∈ 1:n_t
            for i ∈ pump_idx
                cost = cost - Δk*c_elec[k]*9.81*(q⁺[i, k]/1000)*(pump_A[findfirst(x -> x == i, pump_idx)]*(q⁺[i, k]/1000)^2 + pump_B[findfirst(x -> x == i, pump_idx)]*(q⁺[i, k]/1000) + pump_C[findfirst(x -> x == i, pump_idx)])/pump_η
            end
        end =#
        @objective(model, Min, sum(-Δk/3600*c_elec[k]*9.81*(q⁺[i, k]/1000)*θ⁺[i, k]/pump_η for i ∈ pump_idx, k ∈ 1:n_t)) 
        #= from the Matlab code:
        Dh_pumps = -(ap_quad.*q_pumps.^2 + bp_quad.*q_pumps + cp_quad); % Calculate the pump discharge heads.
        efficiencies = 0.78; 
        cost = dth * tou * (9.81*q_pumps.*Dh_pumps) ./ efficiencies; 
        cost = sum(cost); % Calculate total cost.=#
    else
        @objective(model, Min, 0.0)
    end
    # @objective(model, Min, sum(q[i, k] for i ∈ pump_idx, k ∈ 1:n_t))




    ### warm start optimization problem
    if warm_start
        h_j_0, h_tk_0, q_0, q⁺_0, q⁻_0, z_0, θ_0, θ⁺_0, θ⁻_0, u_m_0 = get_starting_point(network, opt_params)
        set_start_value.(h_j, h_j_0)
        # set_start_value.(q, q_0)
        set_start_value.(q⁺, q⁺_0)
        set_start_value.(q⁻, q⁻_0)
        # set_start_value.(z, z_0)
        # set_start_value.(θ, θ_0)
        # set_start_value.(θ⁺, θ⁺_0)
        # set_start_value.(θ⁻, θ⁻_0)
        # set_start_value.(u_m, u_m_0)
        if !isempty(tank_idx)
            set_start_value.(h_tk, h_tk_0)
        end
    end

    ### solve optimization problem
    optimize!(model)
    solution_summary(model)
    term_status = termination_status(model)
    print(term_status)
    accepted_status = [LOCALLY_SOLVED; ALMOST_LOCALLY_SOLVED; OPTIMAL; ALMOST_OPTIMAL; TIME_LIMIT]
    
    if term_status ∉ accepted_status

        @error "Optimization problem did not converge. Please check the model formulation and solver settings."

        if termination_status(model) ∈ [INFEASIBLE; INFEASIBLE_OR_UNBOUNDED]
            
            println("Model is infeasible. Computing IIS...")
            compute_conflict!(model)

            # print conflicting constraints
            #conflict_constraint_list = ConstraintRef[]
            for (F, S) in list_of_constraint_types(model)
                for con in all_constraints(model, F, S)
                    if MOI.get.(model, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                        #push!(conflict_constraint_list, con)
                        println(con)
                    end
                end
            end
            
            #= print confilcting bounds
            #conflicting_bounds = []
            for var in all_variables(model)
                if MOI.get(model, MOI.VariableConflictStatus(), var) == MOI.IN_CONFLICT
                    #push!(conflicting_bounds, ("LowerBound", var))
                    println(var)
                end
            end =#
            
        end

        return OptResults(
            q=zeros(n_l, n_t),
            h_j=zeros(n_j, n_t),
            h_tk=zeros(n_tk, n_t),
            # u_m=zeros(n_l, n_t),
            q⁺=zeros(n_l, n_t),
            q⁻=zeros(n_l, n_t),
            # s=zeros(n_l, n_t),
            θ=zeros(n_l, n_t),
            θ⁺=zeros(n_l, n_t),
            θ⁻=zeros(n_l, n_t),
            z=zeros(n_l, n_t)
        )
    end

    if term_status == TIME_LIMIT
        @error "Maximum time limit reached. Returning best incumbent."
    end

    ### extract results`
    return OptResults(
        q=value.(q),
        h_j=value.(h_j),
        h_tk=value.(h_tk),
        u_m=zeros(n_m,n_t), #value.(u_m),
        q⁺=value.(q⁺),
        q⁻=value.(q⁻),
        s=zeros(n_l,n_t), #value.(s),
        θ=value.(θ),
        θ⁺=value.(θ⁺),
        θ⁻=value.(θ⁻),
        z=value.(z)
    )=#
end=#