##### Import packages and load network data #####
import wntr
import os
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import sys

# set network name and directory
if len(sys.argv) > 1:
    net_name = sys.argv[1]
else:
    net_name = "demo" # "Threenode", "Net1", "Net3", "demo"

print(f"Reading network data for {net_name}...")

inp_file =  net_name + '.inp'
data_path = os.path.join(os.getcwd() + "/" + net_name + "/")


# call wntr and simulate network hydraulics
wn = wntr.network.WaterNetworkModel(os.path.join(data_path, inp_file))
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()


##### Get network and simulation data #####
nt = int(wn.options.time.duration / wn.options.time.hydraulic_timestep)
nt = nt if nt>0 else 1
net_info = dict(
    n_l=wn.num_links,
    n_v=wn.num_valves,
    n_p=wn.num_pipes,
    n_m=wn.num_pumps,
    n_n=wn.num_nodes,
    n_j=wn.num_junctions,
    n_r=wn.num_reservoirs,
    n_tk=wn.num_tanks,
    n_t=nt,
    headloss=wn.options.hydraulic.headloss,
    # units=wn.options.hydraulic.inpfile_units,
    junction_names=wn.junction_name_list,
    reservoir_names=wn.reservoir_name_list,
    tank_names=wn.tank_name_list,
    valve_names=wn.valve_name_list,
    pipe_names=wn.pipe_name_list,
    prv_names=wn.prv_name_list,
    pump_names=wn.pump_name_list
)

# write net_info to json file
with open(os.path.join(data_path, 'net_info.json'), 'w') as f:
    json.dump(net_info, f)



##### Get link data #####
link_df = pd.DataFrame(
    index=pd.RangeIndex(net_info['n_p']),
    columns=['link_id', 'link_type', 'diameter', 'length', 'n_exp', 'C', 'status', 'node_out', 'node_in'],
) # NB: 'C' denotes roughness or HW coefficient for pipes and local (minor) loss coefficient for valves

if net_info['headloss'] == 'H-W':
    n_exp = 1.852
elif net_info['headloss'] == 'D-W':
    n_exp = 2

def link_dict(link):
    if isinstance(link, wntr.network.Pipe):  # check if the link is a pipe

        if link.check_valve == True:
            status = "CV"
        else:
            status = link.initial_status

        return dict(
            link_id=link.name,
            link_type='pipe',
            diameter=link.diameter,
            length=link.length,
            n_exp=n_exp,
            C=link.roughness,
            status=status,
            node_out=link.start_node_name,
            node_in=link.end_node_name
        )
    elif isinstance(link, wntr.network.Valve): # check if the link is a valve
        return dict(
            link_id=link.name,
            link_type='valve',
            diameter=link.diameter,
            length=2*link.diameter,
            n_exp=2,
            C=link.minor_loss,
            status=link.initial_status,
            node_out=link.start_node_name,
            node_in=link.end_node_name
        )
    elif isinstance(link, wntr.network.Pump): # check if the link is a pump

        if link.initial_status == 1:
            status = "Open"
        else:
            status = "Closed"

        return dict(
            link_id=link.name,
            link_type='pump',
            diameter=1,
            length=1e-4,
            n_exp=2,
            C=1e-4,
            status=status,
            node_out=link.start_node_name,
            node_in=link.end_node_name
        )
    
for idx, link in enumerate(wn.links()):
    link_df.loc[idx] = link_dict(link[1])

# write link_df to csv file
link_df.index = link_df.index + 1
link_df.index.names = ['index']
link_df.to_csv(os.path.join(data_path, "link_data.csv"))




##### Get node data #####
node_df = pd.DataFrame(
    index=pd.RangeIndex(wn.num_nodes), columns=["node_id", "node_type", "elev", "xcoord", "ycoord"]
)

def node_dict(node):
    if isinstance(node, wntr.network.elements.Reservoir):
        elev = 0
        node_type = "reservoir"
    elif isinstance(node, wntr.network.elements.Tank):
        elev = node.elevation
        node_type = "tank"
    else:
        elev = node.elevation
        node_type = "junction"

    return dict(
        node_id=node.name,
        node_type=node_type,
        elev=elev,
        xcoord=node.coordinates[0],
        ycoord=node.coordinates[1]
    )

for idx, node in enumerate(wn.nodes()):
    node_df.loc[idx] = node_dict(node[1])

# write node_df to csv file
node_df.index = node_df.index + 1
node_df.index.names = ['index']
node_df.to_csv(os.path.join(data_path, "node_data.csv"))




##### Get tank data #####
tank_df = pd.DataFrame(
    index=pd.RangeIndex(wn.num_tanks), columns=["node_id", "elev", "initial_level", "min_level", "max_level", "diameter"]
)

def tank_dict(tank):

    return dict(
        node_id=tank.name,
        elev=tank.elevation,
        initial_level=tank.init_level,
        min_level=tank.min_level,
        max_level=tank.max_level,
        diameter=tank.diameter,
    )

for idx, tank in enumerate(wn.tanks()):
    tank_df.loc[idx] = tank_dict(tank[1])

# write tank_df to csv file
tank_df.index = tank_df.index + 1
tank_df.index.names = ['index']
tank_df.to_csv(os.path.join(data_path, "tank_data.csv"))




##### Get pump curve data #####
pump_df = pd.DataFrame(
    index=pd.RangeIndex(wn.num_pumps), columns=["link_id", "A", "B", "C"]
)

def pump_dict(pump):

    pump_curve = pump.get_pump_curve()
    pump_curve_points = pump_curve.points
    pump_curve_name = pump_curve.name

    if len(pump_curve_points) == 1:
        print(f"Curve '{pump_curve_name}' is a single-point pump curve. Assigning maximum and low flow points as per EPANET manual.")
        single_point = pump_curve_points[0]
        low_flow_point = (0, 1.33 * single_point[1])
        design_flow_point = single_point
        max_flow_point = (2 * single_point[0], 0)
        pump_curve_points = [low_flow_point, design_flow_point, max_flow_point]

    elif len(pump_curve_points) == 2:
        print(f"Curve '{pump_curve_name}' is a two-point pump curve. Assigning low flow point as per EPANET manual.")
        design_head = max(pump_curve_points, key=lambda t: t[1])[1]
        low_flow_point = (0, 1.33 * design_head)
        pump_curve_points = [low_flow_point, pump_curve_points[0], pump_curve_points[1]]

    # fit quadratic curve to pump curve points
    print(pump_curve_points)
    x_data = np.array([point[0] for point in pump_curve_points])
    y_data = np.array([point[1] for point in pump_curve_points])
    constants = np.poly1d(np.polyfit(x_data, y_data, 2))
    A, B, C = constants

    return dict(
        link_id=pump.name,
        A=A,
        B=B,
        C=C,
    )

for idx, pump in enumerate(wn.pumps()):
    pump_df.loc[idx] = pump_dict(pump[1])

# write pump_df to csv file
pump_df.index = pump_df.index + 1
pump_df.index.names = ['index']
pump_df.to_csv(os.path.join(data_path, "pump_data.csv"))




##### Get demand data #####
demand_df = results.node['demand'].T
col_names = [f'demand_{t}' for t in range(1, len(demand_df.columns)+1)]
demand_df.columns = col_names
demand_df.reset_index(drop=False, inplace=True)
demand_df = demand_df.rename(columns={'name': 'node_id'})

# delete reservoir nodes
demand_df = demand_df[~demand_df['node_id'].isin(net_info['reservoir_names'])]

# delete tank nodes
demand_df = demand_df[~demand_df['node_id'].isin(net_info['tank_names'])]

# delete last time step if greater than 24-hour period
if int(wn.options.time.duration) % 24 == 0:
    demand_df = demand_df.iloc[:, :-1] # delete last time step

# write demand_df to csv file
demand_df.index = demand_df.index + 1
demand_df.index.names = ['index']
demand_df.to_csv(os.path.join(data_path, "demand_data.csv"), index=False)





##### Get boundary head data #####
h0_df = results.node['head'].T
col_names = [f'h0_{t}' for t in range(1, len(h0_df.columns)+1)]
h0_df.columns = col_names
h0_df.reset_index(drop=False, inplace=True)
h0_df = h0_df.rename(columns={'name': 'node_id'})

# only reservoir nodes
h0_df = h0_df[h0_df['node_id'].isin(net_info['reservoir_names'])]

# delete last time step if greater than 24-hour period
if int(wn.options.time.duration) % 24 == 0:
    h0_df = h0_df.iloc[:, :-1] # delete last time step

# write h0_df to csv file
h0_df.index = h0_df.index + 1
h0_df.index.names = ['index']
h0_df.to_csv(os.path.join(data_path, "h0_data.csv"), index=False)





