from filereader import extract_index_dict
import numpy as np
import sklearn
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from filereader import search,search_total_raw,get_filted_data
search_list = ("T04", "T05", "T06", "T02", "T01","T07","T08","T09","T10","T11","T03")

search_list = ("T11",)
# search_list = ("T04","T01")
search_dir = os.path.join(os.path.dirname(os.getcwd()),"data")
# search_dir = r"C:\repo2021\xyrepo\data"
cutter_properties = {}
if os.path.exists("summary/cutter_properties.json"):
    with open("summary/cutter_properties.json","r") as f:
        data = json.load(f)
        cutter_properties.update(data)
appendixs = {}
tool_path = search_total_raw(search_dir,search_list,cutter_pro=cutter_properties)
for k,v in tool_path.items():
    try:
        cutter_properties[k].update({"total":len(v)})
    except:
        pass
result = get_filted_data(tool_path)

with open("summary/cutter_properties.json","w") as f:
    json.dump(cutter_properties,f)
for k,v, in result.items():
    v.to_csv(os.path.join("filted_basic_info",k+".csv"))

fig, ax = plt.subplots(len(search_list), 1, figsize=(10,len(search_list)*5 ))

for i,(k,v) in enumerate(result.items()):
    if isinstance(ax,np.ndarray):
        ax[i].scatter(list(range(v.shape[0])), y=v["mean"])
        ax[i].set_title(r"$\bf{" + k + "-AP " + cutter_properties.get(k)["AP"] + "}$")
    else:
        ax.scatter(list(range(v.shape[0])), y=v["mean"])
        ax.set_title(r"$\bf{" + k + "-AP " + cutter_properties.get(k)["AP"] + "}$")
plt.savefig(f"summary/mean-time-{'-'.join([x for x in search_list])}.png")

plt.show()


   








