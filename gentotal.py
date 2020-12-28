import glob,os,sys
import matplotlib.pyplot as plt
import re
from filereader import get_basic_infos
import pandas as pd
import numpy as np
import datetime

def search(root,targets,reuslt,exclude=None,prefix=None):
    if not isinstance(targets,list):
        targets = [targets]
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            # print('[-]', path)
            search(path, targets,reuslt,exclude,prefix)
        # elif item.find(target) != -1:
        #     print('[+]', path)
        elif any([re.search(target,item) for target in targets]):
            if exclude and item.endswith(exclude):
                continue
            if prefix and item.find(prefix) == -1:
                continue
#             print('[+]', path)
            reuslt.append(path)


test_t = [["T03","T0999"],"T04","T05","T06","T02","T01"]
data_dir = "./data"
data = {}
for t in test_t:
    temp = []
    search(r"D:\xy-repo\data",t,temp,exclude=".png",prefix="load")
    if isinstance(t,str):
        data[t] = temp
    else:
        data[t[0]] = temp
ll = len(data.keys())
fig,ax = plt.subplots(len(test_t),1,figsize=(9,10))
sums = {}

def mean10(df):
    m10 = []
    for i in range(len(df["mean"])):
        m10.append(np.mean(df["mean"][i-10:i])) if i -10 >=0 else m10.append(0)
    return m10

config = {
    "T02":{
        "diff":15
    },
    "T03":{
        "diff":150
    }
}
dif_ratio = 0.1
for i,(k,v) in enumerate(data.items()):
    t = get_basic_infos(data[k])
    df = pd.DataFrame(t)
    df.sort_values(by=["index"],inplace=True)
    df["mean10"] = mean10(df)
    diff = config.get(k,{}).get("diff",0)
    if diff:
        df = df.loc[np.where((df["mean"] - df["mean10"])/df["mean10"] < -dif_ratio,False,True)]
    sums[k] = df["sum"]
    ax[i].scatter(list(range(len(df["mean"]))),y=df["mean"])
    ax[i].set_title(k)


import time
plt.savefig(f"summary-{dif_ratio}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.png")
plt.show()