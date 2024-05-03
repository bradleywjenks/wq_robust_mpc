"""
Collection of functions for loading hydraulic model and creating network data structure
"""

using InlineStrings
using Parameters
using SparseArrays
using LinearAlgebra
using AutomaticDocstrings
using DataFrames
using CSV, JSON

const NETWORK_PATH = pwd() * "/networks/"


@with_kw mutable struct Network
    name::String
    dir::String
    n_l::Int64
    n_v::Int64
    n_p::Int64
    n_m::Int64
    n_n::Int64
    n_j::Int64
    n_r::Int64
    n_tk::Int64
    n_t::Int64
    A10_res::SparseArrays.SparseMatrixCSC{Int64,Int64}
    A10_tank::SparseArrays.SparseMatrixCSC{Int64,Int64}
    A12::SparseArrays.SparseMatrixCSC{Int64,Int64}
    d::Matrix{Float64}
    h0::Matrix{Float64}
    L::Vector{Float64}
    C::Vector{Float64}
    D::Vector{Float64}
    nexp::Vector{Float64}
    r::Vector{Float64}
    status::Vector{String}
    elev::Vector{Float64}
    tank_init::Vector{Float64}
    tank_min::Vector{Float64}
    tank_max::Vector{Float64}
    tank_area::Vector{Float64}
    pump_A::Vector{Float64}
    pump_B::Vector{Float64}
    pump_C::Vector{Float64}
    simtype::String3
    pipe_idx::Vector{Int64}
    valve_idx::Vector{Int64}
    pump_idx::Vector{Int64}
    junction_idx::Vector{Int64}
    reservoir_idx::Vector{Int64}
    tank_idx::Vector{Int64}
    prv_names::Vector{Union{String,Int64}}
    X_coord::Vector{Float64}
    Y_coord::Vector{Float64}
    link_name_to_idx::Dict{Union{String31,String15,String7,Int64},Int64}
    node_name_to_idx::Dict{Union{String31,String15,String7,Int64},Int64}
    link_names::Vector{Union{String31,String15,String7,Int64}}
    node_names::Vector{Union{String31,String15,String7,Int64}}
    A21::SparseArrays.SparseMatrixCSC{Int64,Int64} = A12'
    A = hcat(A12, A10_res, A10_tank)
    B_prv::Vector{Int64}
    B_pump::Vector{Int64}
    @assert simtype == "H-W" "simtype must be H-W, it is $simtype"
end

Base.copy(n::Network) = Network(
    name=deepcopy(n.name),
    dir=deepcopy(n.dir),
    n_l=deepcopy(n.n_l),
    n_v=deepcopy(n.n_v),
    n_p=deepcopy(n.n_p),
    n_m=deepcopy(n.n_m),
    n_n=deepcopy(n.n_n),
    n_j=deepcopy(n.n_j),
    n_r=deepcopy(n.n_r),
    n_tk=deepcopy(n.n_tk),
    n_t=deepcopy(n.n_t),
    A10_res=deepcopy(n.A10_res),
    A10_tank=deepcopy(n.A10_tank),
    A12=deepcopy(n.A12),
    d=deepcopy(n.d),
    h0=deepcopy(n.h0),
    L=deepcopy(n.L),
    C=deepcopy(n.C),
    D=deepcopy(n.D),
    nexp=deepcopy(n.nexp),
    r=deepcopy(n.r),
    status=deepcopy(n.status),
    elev=deepcopy(n.elev),
    tank_init=deepcopy(n.tank_init),
    tank_min=deepcopy(n.tank_min),
    tank_max=deepcopy(n.tank_max),
    tank_area=deepcopy(n.tank_area),
    pump_A=deepcopy(n.pump_A),
    pump_B=deepcopy(n.pump_B),
    pump_C=deepcopy(n.pump_C),
    simtype=deepcopy(n.simtype),
    pipe_idx=deepcopy(n.pipe_idx),
    valve_idx=deepcopy(n.valve_idx),
    pump_idx=deepcopy(n.pump_idx),
    junction_idx=deepcopy(n.junction_idx),
    tank_idx=deepcopy(n.tank_idx),
    reservoir_idx=deepcopy(n.reservoir_idx),
    prv_name=deepcopy(n.prv_name),
    X_coord=deepcopy(n.X_coord),
    Y_coord=deepcopy(n.Y_coord),
    link_name_to_idx=deepcopy(n.link_name_to_idx),
    node_name_to_idx=deepcopy(n.node_name_to_idx),
    link_names=deepcopy(n.link_names),
    node_names=deepcopy(n.node_names),
    A21=deepcopy(n.A21),
    B_prv=deepcopy(n.B_prv),
    B_pump=deepcopy(n.B_pump))



""" 
Make network data.
"""
function load_network(net_name::String, network_dir=NETWORK_PATH * net_name; dbv_name=nothing, iv_name=nothing, afv_name=nothing)
    
    @info "Loading $net_name network data."
    if network_dir[end] != "/"
        network_dir *= "/"
    end

    netinfo = JSON.parsefile(network_dir * "net_info.json")
    linkinfo = CSV.read(network_dir * "link_data.csv", DataFrame)
    nodeinfo = CSV.read(network_dir * "node_data.csv", DataFrame)
    tankinfo = CSV.read(network_dir * "tank_data.csv", DataFrame)
    pumpinfo = CSV.read(network_dir * "pump_data.csv", DataFrame)
    demands = CSV.read(network_dir * "demand_data.csv", DataFrame)
    demands[:, 2:end]
    d = Array{Float64}(demands[:, 2:end])
    d = 1e3 .* d # demands converted to L/s
    h0_data = CSV.read(network_dir * "h0_data.csv", DataFrame)
    h0 = Array{Float64}(h0_data[:, 2:end])
    n_l = netinfo["n_l"]
    n_v = netinfo["n_v"]
    n_p = netinfo["n_p"]
    n_m = netinfo["n_m"]
    n_n = netinfo["n_n"]
    n_j = netinfo["n_j"]
    n_r = netinfo["n_r"]
    n_tk = netinfo["n_tk"]
    n_t = netinfo["n_t"]
    simtype = netinfo["headloss"]

    # element physical properties
    L = linkinfo.length
    D = linkinfo.diameter
    C = linkinfo.C
    status = linkinfo.status
    XCoord = nodeinfo.xcoord
    YCoord = nodeinfo.ycoord
    nexp = linkinfo.n_exp
    elev = nodeinfo.elev
    tank_init = tankinfo.initial_level
    tank_min = tankinfo.min_level
    tank_max = tankinfo.max_level
    tank_area = π .* (tankinfo.diameter ./ 2) .^ 2
    pump_A = pumpinfo.A
    pump_B = pumpinfo.B
    pump_C = pumpinfo.C

    # link names and indices
    link_names = linkinfo.link_id
    link_idx = linkinfo.index
    pipe_idx = findall(linkinfo.link_type .== "pipe")
    valve_idx = findall(linkinfo.link_type .== "valve")
    pump_idx = findall(linkinfo.link_type .== "pump")

    # node names and indices
    node_names = nodeinfo.node_id
    node_idx = nodeinfo.index
    junction_idx = findall(nodeinfo.node_type .== "junction")
    reservoir_idx = findall(nodeinfo.node_type .== "reservoir")
    tank_idx = findall(nodeinfo.node_type .== "tank")

    link_name_to_idx = Dict(link_names .=> link_idx)
    node_name_to_idx = Dict(node_names .=> node_idx)

    # adjacency matrices
    node_in_idx = [node_name_to_idx[name] for name in linkinfo.node_in]
    node_out_idx = [node_name_to_idx[name] for name in linkinfo.node_out]
    iA = zeros(Int64, 2n_l)
    jA = zeros(Int64, 2n_l)
    vA = zeros(Int64, 2n_l)
    ii = 1
    for k in 1:n_l
        iA[ii] = k
        jA[ii] = node_in_idx[k]
        vA[ii] = 1
        ii = ii + 1
        iA[ii] = k
        jA[ii] = node_out_idx[k]
        vA[ii] = -1
        ii = ii + 1
    end
    A = sparse(iA, jA, vA)
    A12 = A[:, 1:n_j]
    A10_res = A[:, n_j+1:n_j+n_r]
    A10_tank = A[:, n_j+n_r+1:n_j+n_r+n_tk]


    # assign pressure control valve links (if any)
    prv_names = netinfo["prv_names"]
    prv_idx = [link_name_to_idx[parse(Int, name)] for name in prv_names]
    B_prv = zeros(Int64, n_l)
    B_prv[prv_idx] .= 1
    B_pump = zeros(Int64, n_l)
    B_pump[pump_idx] .= 1

    # compute resistance coefficients
    r = 10.67 .* L ./ ((C .^ nexp) .* (D .^ 4.871))
    if ~isnothing(valve_idx) && ~isempty(valve_idx)
        r[valve_idx] = (8 / (pi^2 * 9.81)) .* (D[valve_idx] .^ -4) .* C[valve_idx]
    elseif ~isnothing(pump_idx) && ~isempty(pump_idx)
        r[pump_idx] .= 1e-4
    end
    r = ((1e-3) .^ nexp) .* r # conversion to flows in l/s

    network = Network(name=net_name, dir=network_dir, n_l=n_l, n_p=n_p, n_v=n_v, n_m=n_m, n_n=n_n, n_j=n_j, n_r=n_r, n_tk=n_tk, n_t=n_t, A10_res=A10_res, A10_tank=A10_tank, A12=A12, B_prv=B_prv, B_pump=B_pump, d=d, h0=h0, L=L, C=C, D=D, status=status, nexp=nexp, tank_init=tank_init, tank_min=tank_min, tank_max=tank_max, tank_area=tank_area, pump_A=pump_A, pump_B=pump_B, pump_C=pump_C, r=r, elev=elev, simtype=simtype, X_coord=XCoord, Y_coord=YCoord, link_name_to_idx=link_name_to_idx, node_name_to_idx=node_name_to_idx, link_names=link_names, node_names=node_names, pipe_idx=pipe_idx, valve_idx=valve_idx, pump_idx=pump_idx, junction_idx=junction_idx, tank_idx=tank_idx, reservoir_idx=reservoir_idx, prv_names=prv_names);

    return network
end
