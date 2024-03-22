    # Load the JSON data
from pyvis.network import Network
net = Network(height='830px')

import json
file_path = 'training_scripts/fluids/result.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# 1400 in height in total.
# 2430 in width in total.

phase = 0
number_of_actions = 16
for phase in range(data["phase"]+1):
    for i in range(number_of_actions):
        node_key = f"{phase}-a{i}"
        prev_node_key = f"{phase-1}-a{i}"
        node_id = 100*phase+i
        if node_key in data:
            color = "#66ff00" if data[node_key] == "s" else "#AED6F1"
        else:
            color = "#a8adbe"
        
        if phase == 0 or prev_node_key in data:
            net.add_node(node_id, label=f"{i:02d}", x=[30*i], y=[60*phase], shape="circle", color=color)
            if phase > 0 and (node_key in data or prev_node_key in data):
                net.add_edge(100*(phase-1)+i, node_id)
    phase = phase + 1
net.toggle_physics(False)
net.toggle_drag_nodes(False)
net.show('result.html')