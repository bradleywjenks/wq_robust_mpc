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
    x_wq_0=deepcopy(o.x_wq_0)
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
function make_prob_data(network::Network, Δt, Δk, sim_days, disc_method; pmin::Int64=15, pmax::Int64=200, Qmax_mul=1000, Qmin_mul=-1000, x_wq_bounds=(0.5, 3), u_wq_bounds=(0, 5), QA=false, quadratic_approx="relative", J=nothing, kb=0.5, kw=0, x_wq_0=0, obj_type="AZP")

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
        θmin .= (network.r .* Qmax .* abs.(Qmax) .^ (network.nexp .- 1)) .* -1
        θmax .= network.r .* Qmax .* abs.(Qmax) .^ (network.nexp .- 1)
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
        x_wq_0=x_wq_0
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
function optimize_hydraulic_wq(network::Network, opt_params::OptParams; x_wq_0=0.5, solver="Gurobi", integer=true, warm_start=false, heuristic=false)

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
    core_links = network.core_links

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
    Δx_p = opt_params.Δx_p
    s_p = opt_params.s_p
    n_s = sum(s_p)
    kb = (opt_params.kb/3600/24) # units = 1/second

    # simulation times
    T = opt_params.T
    k_t = opt_params.k_t
    n_t = length(k_t)
    Δt = opt_params.Δt
    Δk = opt_params.Δk

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

    if solver == "Gurobi"
        
        ### GUROBI
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "TimeLimit", 21600) # 6-hour time limit
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
        set_optimizer_attribute(model, "max_iter", 3000)
        set_optimizer_attribute(model, "warm_start_init_point", "yes")
        # set_optimizer_attribute(model, "linear_solver", "ma57")
        # set_optimizer_attribute(model, "linear_solver", "spral")
        set_optimizer_attribute(model, "linear_solver", "ma97")
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

    # hydrualic
    @variable(model, Hmin_j[i, k] ≤ h_j[i=1:n_j, k=1:n_t] ≤ Hmax_j[i, k])
    @variable(model, Hmin_tk[i, k] ≤ h_tk[i=1:n_tk, k=1:n_t+1] ≤ Hmax_tk[i, k])
    @variable(model, Qmin[i, k] ≤ q[i=1:n_l, k=1:n_t] ≤ Qmax[i, k])
    @variable(model, 0 ≤ q⁺[i=1:n_l, k=1:n_t])
    @variable(model, 0 ≤ q⁻[i=1:n_l, k=1:n_t])
    @variable(model, 0 ≤ s[i=1:n_l, k=1:n_t])
    @variable(model, θmin[i, k] ≤ θ[i=1:n_l, k=1:n_t] ≤ θmax[i, k])
    @variable(model, θ⁺[i=1:n_l, k=1:n_t])  
    @variable(model, θ⁻[i=1:n_l, k=1:n_t])
    @variable(model, u_m[i=1:n_m, k=1:n_t])

    if solver ∈ ["Gurobi", "SCIP"]  && integer
        @variable(model, z[i=1:n_l, k=1:n_t], binary=true)
    elseif solver ∈ ["Gurobi", "SCIP"] && !integer
        @variable(model, 0 ≤ z[i=1:n_l, k=1:n_t] ≤ 1)
        # @constraint(model, complementarity[i=1:n_l, k=1:n_t], z[i, k] * (1 - z[i, k]) == 0.0)
    else
        @variable(model, 0 ≤ z[i=1:n_l, k=1:n_t] ≤ 1)
        # @constraint(model, flow_complementarity[i=1:n_l, k=1:n_t], z[i, k] * (1 - z[i, k]) ≤ 1e-4)
    end


    # water quality
    # TO BE COMPLETED!!!

    # ### fix variables
    for i ∈ pump_idx
        fix.(θ⁻[i, :], 0.0, force=true)
        fix.(q⁻[i, :], 0.0, force=true)
    end

    ### define constraints

    # initial conditions
    for i ∈ tank_idx
        fix.(h_tk[findfirst(x -> x == i, tank_idx), 1], tk_init[findfirst(x -> x == i, tank_idx)]; force=true)
    end

    # engergy conservation
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
            @constraint(model, θ⁺[i, k] == -1 .* (pump_A[findfirst(x -> x == i, pump_idx)] * (q⁺[i, k] / 1000)^2 + pump_B[findfirst(x -> x == i, pump_idx)] * (q⁺[i, k] / 1000) + pump_C[findfirst(x -> x == i, pump_idx)]) + u_m[findfirst(x -> x == i, pump_idx), k])
        end
    end
        

    # flow and head loss direction constraints
    @constraint(model, flow_value[i=1:n_l, k=1:n_t], q⁺[i, k] - q⁻[i, k] == q[i, k])
    @constraint(model, flow_value_abs[i=1:n_l, k=1:n_t], q⁺[i, k] + q⁻[i, k] == s[i, k])
    @constraint(model, head_loss_value[i=1:n_l, k=1:n_t], θ⁺[i, k] - θ⁻[i, k] == θ[i, k])


    # complementarity constraints for flow direction
    if solver ∈ ["Gurobi", "SCIP"]
        # @constraint(model, flow_direction_pos[i=core_links, k=1:n_t], q⁺[i, k] ≤ z[i, k] * Qmax[i, k])
        # @constraint(model, flow_direction_neg[i=core_links, k=1:n_t], q⁻[i, k] ≤ (1 - z[i, k]) * abs(Qmin[i, k]))
        @constraint(model, flow_direction[i=1:n_l, k=1:n_t], [q⁻[i, k], q⁺[i, k]] in SOS1())
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

    # complementarity constraints for pump status
    if !isempty(pump_idx)
        if solver == "Gurobi"
            @constraint(model, pump_status[i=pump_idx, k=1:n_t], [u_m[findfirst(x -> x == i, pump_idx), k], q⁺[i, k]] in SOS1())
            # @constraint(model, pump_status[i=1:n_l, k=1:n_t],  u_m[i, k] * q⁺[i, k] ≤ 1e-6)
            # @constraint(model, pump_status_1[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≤ 1e-6)
            # @constraint(model, pump_status_2[i=pump_idx, k=1:n_t],  u_m[findfirst(x -> x == i, pump_idx), k] * q⁺[i, k] ≥ -1e-6)
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

    # water quality...



    ### define objective functions
    # if obj_type == "AZP"
    #     # insert code here...
    # else
    #     # insert code here...
    # end
    if !isempty(pump_idx)
        @objective(model, Min, sum(q[i, k] for i ∈ pump_idx, k ∈ 1:n_t))
    else
        @objective(model, Min, 0.0)
    end





    ### warm start optimization problem
    if warm_start
        h_j_0, h_tk_0, q_0, q⁺_0, q⁻_0, z_0, θ_0, θ⁺_0, θ⁻_0, u_m_0 = get_starting_point(network, opt_params)
        set_start_value.(h_j, h_j_0)
        # set_start_value.(q, q_0)
        set_start_value.(q⁺, q⁺_0)
        set_start_value.(q⁻, q⁻_0)
        set_start_value.(z, z_0)
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
    accepted_status = [LOCALLY_SOLVED; ALMOST_LOCALLY_SOLVED; OPTIMAL; ALMOST_OPTIMAL]
    
    if term_status ∉ accepted_status

        @error "Optimization problem did not converge. Please check the model formulation and solver settings."

        return OptResults(
            q=zeros(n_l, n_t),
            h_j=zeros(n_j, n_t),
            h_tk=zeros(n_tk, n_t),
            u_m=zeros(n_l, n_t),
            q⁺=zeros(n_l, n_t),
            q⁻=zeros(n_l, n_t),
            s=zeros(n_l, n_t),
            θ=zeros(n_l, n_t),
            θ⁺=zeros(n_l, n_t),
            θ⁻=zeros(n_l, n_t),
            z=zeros(n_l, n_t)
        )
    end


    ### extract results`
    return OptResults(
        q=value.(q),
        h_j=value.(h_j),
        h_tk=value.(h_tk),
        u_m=value.(u_m),
        q⁺=value.(q⁺),
        q⁻=value.(q⁻),
        s=value.(s),
        θ=value.(θ),
        θ⁺=value.(θ⁺),
        θ⁻=value.(θ⁻),
        z=value.(z)
    )
end
