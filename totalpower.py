import glob,os,sys
import matplotlib.pyplot as plt
import re
from filereader import get_basic_infos,search
import pandas as pd
import json
#
# def search(root,targets,reuslt,exclude=None,prefix=None):
#     if not isinstance(targets,list):
#         targets = [targets]
#     items = os.listdir(root)
#     for item in items:
#         path = os.path.join(root, item)
#         if os.path.isdir(path):
#             # print('[-]', path)
#             search(path, targets,reuslt,exclude,prefix)
#         # elif item.find(target) != -1:
#         #     print('[+]', path)
#         elif any([re.search(target,item) for target in targets]):
#             if exclude and item.endswith(exclude):
#                 continue
#             if prefix and item.find(prefix) == -1:
#                 continue
# #             print('[+]', path)
#             reuslt.append(path)


test_t = ["T03","T04","T05","T06","T01","T07","T09","T10","T11","T02","T01",'T08']
# test_t = ['T07','T09']

data_dir = "./data"
data = {}
for t in test_t:
    temp = []
    search(r"C:\repo2021\xyrepo\data",(t,),temp,exclude=".png",prefix="load")
    if isinstance(t,str):
        data[t] = temp
    else:
        data[t[0]] = temp


infos = {}


ll = len(data.keys())
fig,ax = plt.subplots(4,1,figsize=(9,10))
sums = {}
powbystep = {}
for i,(k,v) in enumerate(data.items()):
    df = get_basic_infos(data[k])
    df = pd.DataFrame(df)
    df.sort_values(by=["index"],inplace=True)
    infos[k] = {"search_counts":len(v),"vlid_counts":df.shape[0]}
    sums[k] = df["sum"]
    powbystep[k] = df["sum"].to_list()
    # if k == "T02":
    #     sums["index"] = df["index"]
infodf = pd.DataFrame(infos)
with open("power.json",'w') as f:
    json.dump(powbystep,f)
infodf.to_csv("info_of_length.csv")
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in sums.items()]))
# df = pd.DataFrame(sums)
df.fillna(value=0,inplace=True)
# df.sort_values(by=["index"],inplace=True)
df.to_csv("data-base-7.csv")

