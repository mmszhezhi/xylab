import joblib
import pandas as pd
import numpy as np
from filereader import vib_read_raw, extract_index_dict
import glob, os
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

    return result, sum(selected.index * (selected['ratio'] / sum(selected['ratio'])))


def get_property(raw):
    lofl = {}
    ret = {}
    for k, v in raw.items():
        lofl[str(k)] = len(v["vibration-x"])
        ret[k] = np.mean(v["vibration-x"])
    t = sorted(lofl.items(), key=lambda x: x[1])
    minl = t[0]
    maxl = t[-1]
    df = pd.DataFrame(lofl, index=["value_counts"])
    df = df.transpose()
    return df, df["value_counts"].value_counts(), ret


root = os.getcwd()
weighted_lens = {}

jump = ["T01","T02","T03","T04","T05","T06","T07","T08","T09","T10"]
for tool in tools:
    print(tool)
    index_freq = {}
    name = os.path.basename(tool)
    if name.split(".")[0] in jump:
        print("jump :", name)
        continue
    df = pd.read_csv(tool, index_col=0)
    index_dict = extract_index_dict(df["paths"])
    rawx = vib_read_raw(index_dict, df, ('1'))
    dfindex, prop, mean_dict = get_property(rawx)
    selected, weighted_len = min_max_filter(dfindex, prop)
    weighted_lens[name.split(".")[0]] = weighted_len
    for i in selected.index:
        data = rawx.get(int(i))
        shifted = np.array(data['vibration-x']) - mean_dict.get(int(i))
        index_freq[i] = {"vibration-x": np.abs(np.fft.fft(shifted)[:math.ceil(len(shifted) * 0.5)]).tolist()}
        print(f"{tool}  {i} of {selected.shape[0]}")
    os.makedirs(os.path.join(root, f"index_freq"), exist_ok=True)
    with open(os.path.join(root, f"index_freq/{name.split('.')[0]}.json"), "w") as f:
        json.dump(index_freq, f)
    # break

with open(os.path.join(root, f"index_freq/weighted_len.json"), "w") as f:
    json.dump(weighted_lens, f)
