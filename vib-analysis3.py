import joblib
import pandas as pd
import numpy as np
from filereader import vib_read_raw,extract_index_dict
import glob,os
import json
import math


tools = glob.glob("./filted_basic_info/*")


def min_max_filter(df, prop):
    total = prop.values.sum()
    prop = prop.to_frame()
    prop["ratio"] = prop["value_counts"] / total
    prop["cumlative"] = prop["ratio"].cumsum()
    prop["dev"] = np.abs(prop.index - prop.index[0])
    selected = prop[((prop["cumlative"] < 0.93) | (prop["dev"] <= 500)) & (prop["value_counts"] >= 10)]
    result = df[df["value_counts"].isin(selected.index.values)]
    return result


def get_property(raw):
    lofl = {}
    for k,v in raw.items():
        lofl[str(k)] = len(v["vibration-x"])
    t= sorted(lofl.items(),key=lambda x:x[1])
    minl = t[0]
    maxl = t[-1]
    df = pd.DataFrame(lofl,index=["value_counts"])
    df = df.transpose()
    return df,df["value_counts"].value_counts()



root = os.getcwd()
for tool in tools:
    print(tool)
    index_freq = {}
    name = os.path.basename(tool)
    df = pd.read_csv(tool,index_col=0)
    index_dict = extract_index_dict(df["paths"])
    rawx = vib_read_raw(index_dict,df,('1'))
    dfindex,prop = get_property(rawx)
    selected = min_max_filter(dfindex,prop)
    for i in selected.index:
        data = rawx.get(int(i))
        index_freq[i] = {"vibration-x":np.abs(np.fft.fft(data["vibration-x"])[:math.ceil(len(data["vibration-x"]) *0.5)]).tolist()}
        print(f"{tool}  {i} of {selected.shape[0]}")

    with open(os.path.join(root,f"index_freq/{name.split('.')[0]}.json"),"w") as f:
        json.dump(index_freq,f)