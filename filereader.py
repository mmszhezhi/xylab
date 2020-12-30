files = [r"D:\\xy-repo\\data\\20201214\\192.168.101.10\\T03\\T03_load_100_6000_2000_0.2_steel_20201214_161439",
         r"D:\\xy-repo\\data\\20201214\\192.168.101.10\\T03\\T03_load_104_6000_2000_0.2_steel_20201214_164641"]
import json, os, sys
import numpy as np
import pandas as pd
import math
import logging

import logging
import re

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__file__)
channel_names = {'1': 'vibration-x', '2': 'vibration-y', '3': 'vibration-z', '4': 'sound'}


def get_basic_info(file):
    if not file:
        return
    cat = os.path.basename(file)
    with open(file) as f:
        datas = f.read()
    data = json.loads(datas)
    ret = {}
    data = np.array(data["value"])
    ret["length"] = data.shape
    ret["min"] = data.min()
    ret["max"] = data.max()
    ret["var"] = round(data.var(), 2)
    ret["mean"] = round(data.mean(), 2)
    return ret


def search(root, targets, reuslt, exclude=None, prefix=None, appendix=None):
    """
    search desired file in given directory recurrently,exclude and prefix has a relative higher privilege than targets
    :param appendix: optional container for additional information of files like AP,AE,SPEED...
    :param root: search directory
    :param targets: list of str search keys
    :param reuslt: container of result
    :param exclude: list of str search exclude keys
    :param prefix: prefix of file name
    :return:
    """
    if not isinstance(targets, list):
        targets = [targets]
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            # print('[-]', path)
            search(path, targets, reuslt, exclude, prefix,appendix)
        # elif item.find(target) != -1:
        #     print('[+]', path)
        elif any([re.search(target+"_", item) for target in targets]):
            if exclude and item.endswith(exclude):
                continue
            if prefix and item.find(prefix) == -1:
                continue
            #             print('[+]', path)
            appendixs = item.split("_")
            if appendix is not None and not appendix.get(appendixs[0],None):
                appendix.update({appendixs[0]: {
                    "speed": appendixs[3],
                    "feed": appendixs[4],
                    "AP": appendixs[5]}})
            reuslt.append(path)


def filter(data):
    df = pd.DataFrame({"load": data})
    df["diff"] = abs(df["load"] - df["load"].shift(1, fill_value=df["load"][0]))
    ll = df.shape[0]
    if ll < 92:
        m = np.mean(df["load"][math.ceil(ll * 0.3):math.ceil(ll * 0.7)])
    else:
        m = np.mean(df["load"][math.ceil(ll * 0.5):math.ceil(ll * 0.85)])
    # diff_index = [10,50]
    f = lambda x: 1 / 12 * x
    diff_index = f(m)
    df.reset_index(inplace=True)
    # df["class"] = np.where(((abs(df["diff"] < 5)) &((df["index"] > math.ceil(ll*0.92)) | (df["index"] < math.ceil(ll*0.08))))& (df["load"] > 100), 1, 0)
    # df["class"] = np.where(((abs(df["diff"] < 5)) & ((df["index"] > math.ceil(ll * 0.92)) | (df["index"] < math.ceil(ll * 0.08))) ) & (df["load"] > 100), 1, 0)
    # df["class"] = np.where(((df["index"] > math.ceil(ll * 0.85)) | (df["index"] < math.ceil(ll * 0.15)))&(df["diff"]>20),0,1)
    df["class"] = np.where((df["diff"] > diff_index), 0, 1)
    df["c_diff"] = abs(df["class"] - df["class"].shift(1, fill_value=None))
    starts = np.where(df["c_diff"] == 1)[0]
    if ll < 92:
        starts = starts[np.where(((starts / ll) < 0.15) | ((starts / ll) > 0.8))]
    lengths = [starts[x + 1] - starts[x] for x in range(len(starts) - 1)]
    if not lengths:
        log.debug(f"diff {diff_index} seems to large")
        return None, None, None
    s = lengths.index(max(lengths))
    df["label"] = np.where(df["index"].between(starts[s], starts[s + 1] + 1), 1, 0)
    if df["load"][starts[s]:starts[s + 1] + 1].mean() < 100:
        log.warning("load mean less than 100")
        return None, None, None
    # df = df.iloc[starts[s]:starts[s+1]+1]
    # df = df.iloc[df[df["class"] == 1].index.min():df[df["class"] == 1].index.max()]
    return df, starts[s], starts[s + 1] + 1


def search_total_raw(search_dir, search_list=("T03", "T04", "T05", "T06", "T02", "T01")):
    """
    :return: path of load
    """
    # test_t = [["T03", "T0999"], "T04", "T05", "T06", "T02", "T01"]
    data_dir = "./data"
    data = {}
    for t in search_list:
        temp = []
        search(search_dir, t, temp, exclude=".png", prefix="load")
        if isinstance(t, str):
            data[t] = temp
        else:
            data[t[0]] = temp
    return data


def base_filter(data, toolid=None):
    df = pd.DataFrame({"load": data})
    if df["load"][:3].mean() > 70:
        log.warning(f"{toolid}  mean3 great than 50 {df['load'][:3].mean()}")
        return None, None, None
    df["diff"] = abs(df["load"] - df["load"].shift(2, fill_value=df["load"][0]))
    ll = df.shape[0]
    if ll < 92:
        m = np.mean(df["load"][math.ceil(ll * 0.3):math.ceil(ll * 0.7)])
        minimal = np.min(df["load"][math.ceil(ll * 0.3):math.ceil(ll * 0.7)])
    else:
        m = np.mean(df["load"][math.ceil(ll * 0.5):math.ceil(ll * 0.85)])
        minimal = np.min(df["load"][math.ceil(ll * 0.5):math.ceil(ll * 0.85)])
    # diff_index = [10,50]
    threshold: int = 50
    min_threshold = 45
    if m < threshold or minimal < min_threshold:
        log.warning(f"{toolid} load {m} less than {threshold} or {minimal} less than {min_threshold}")
        return None, None, None
    f = lambda x: 1 / 12 * x
    diff_index = f(m)
    diff_index = 10
    df.reset_index(inplace=True)
    df["class"] = np.where((df["diff"] > diff_index), 1, 0)
    start = df[df["class"] == 1].index.min() + 2
    tail = math.ceil(ll * 0.83)
    head = math.ceil(ll * 0.18)
    end = df[df["class"] == 1].index.max()
    end = df["diff"][tail:].argmax() + tail - 2
    try:
        df["label"] = np.where(df["index"].between(start, end), 1, 0)
        if df["load"][start:end].mean() < threshold or abs(start - end) < 25:
            log.warning(f'{toolid} work load mean less than {threshold} or abs(start-end) less 25 {start}-{end}')
            return None, None, None
    except Exception as e:
        log.warning(repr(e))
        return None, None, None
    return df, start, end


config = {
    "T04": {
        "diff": 10
    }
}


def get_basic_infos(files):
    if not files:
        return
    index = []
    length = []
    mins = []
    max = []
    var = []
    mean = []
    sum = []
    trims = []
    paths = []
    for file in files:
        cat = os.path.basename(file)
        tool_name = cat.split("_")[0]
        i = int(cat.split("_")[2])
        if tool_name == "T02" and i == 8:
            print(tool_name)
        # if i in [111, 114, 115, 116, 117, 118, 119, 120]:
        #     print(i)
        with open(file) as f:
            datas = f.read()
        data = json.loads(datas)
        ret = {}
        data = np.array(data["value"])
        df, s, e = base_filter(data, i)

        if df is None or s == e:
            log.warning(f"basic filted: {tool_name} {i} start-end:{s}-{e}  filter return None or start=end ")
            continue
        trims.append((round(s / df.shape[0], 2), round(e / df.shape[0], 2)))
        df = df.iloc[s:e]

        paths.append(file)
        sum.append(round(df["load"].sum()) - df["load"].shape[0] * 38)
        index.append(i)
        length.append(df["load"].shape[0])
        mins.append(df["load"].min())
        max.append(df["load"].max())
        var.append(round(df["load"].var(), 2))
        mean.append(round(df["load"].mean(), 2))
    ret["index"] = index
    ret["length"] = length
    ret["min"] = mins
    ret["max"] = max
    ret["var"] = var
    ret["mean"] = mean
    ret["sum"] = sum
    ret["trims"] = trims
    ret["paths"] = paths
    return ret


def file_read(src):
    with open(src, "r") as f:
        data = f.read()
    data = json.loads(data)
    return data["value"]


def get_filted_data(data):
    """
    get filted data
    :param data: path of load
    :return:
    """

    def mean10(df):
        m10 = []
        for i in range(len(df["mean"])):
            m10.append(np.mean(df["mean"][i - 10:i])) if i - 10 >= 0 else m10.append(0)
        return m10

    config = {
        "T02": {
            "diff": 15
        },
        "T03": {
            "diff": 150
        }
    }
    dif_ratio = 0.1
    result = {}
    for i, (k, v) in enumerate(data.items()):
        t = get_basic_infos(data[k])
        df = pd.DataFrame(t)
        df.sort_values(by=["index"], inplace=True)
        df["mean10"] = mean10(df)
        diff = config.get(k, {}).get("diff", 0)
        if diff:
            df = df.loc[np.where((df["mean"] - df["mean10"]) / df["mean10"] < -dif_ratio, False, True)]
        result[k] = df
    return result


def extract_index_dict(data: list):
    """
    :param data: path of load
    :return:
    """
    result = {}
    for item in data:
        filename = os.path.basename(item)
        index = filename.split("_")[2]
        result[int(index)] = item
    return result


def vib_read_trime(paths, df: pd.DataFrame, ch: tuple):
    """
    read vibration data with give paths load path
    :param paths: paths of load {}
    :param df: df of tool includ index,trims
    :param ch: chnnel of vibration ("1","2","3")
    :return:
    """
    result = {}
    ll = df.shape[0]
    pointer = 0
    for row in df.iterrows():
        log.warning(f" {pointer} of {ll}")
        pointer += 1
        path_load = paths.get(row[1]["index"])
        if not path_load:
            continue
        if not ch:
            continue
        ret = {}
        for c in ch:
            vib = channel_names.get(c, None)
            vib_path = path_load.replace("load", vib)
            try:
                t = file_read(vib_path)
            except:
                log.debug(f"file not find {vib_path}")
                break
            l = len(t)
            trims = eval(row[1]["trims"])
            t = t[math.ceil(l * trims[0]):math.ceil(l * trims[1])]
            ret[vib] = t
        else:
            result[row[1]["index"]] = ret

    return result


def vib_read_raw(paths, df: pd.DataFrame, ch: tuple):
    """
    read vibration data with give paths load path
    :param paths: paths of load {}
    :param df: df of tool includ index,trims
    :param ch: chnnel of vibration ("1","2","3")
    :return:
    """
    result = {}
    ll = df.shape[0]
    pointer = 0
    for row in df.iterrows():
        log.warning(f" {pointer} of {ll}")
        pointer += 1
        path_load = paths.get(row[1]["index"])
        if not path_load:
            continue
        if not ch:
            continue
        ret = {}
        for c in ch:
            vib = channel_names.get(c, None)
            vib_path = path_load.replace("load", vib)
            try:
                t = file_read(vib_path)
            except:
                log.debug(f"file not find {vib_path}")
                break
            ret[vib] = t
        else:
            result[row[1]["index"]] = ret

    return result


t04 = [r'D:\\xy-repo\\data\\20201217\\192.168.101.10\\T04\\T04_load_108_6000_2000_0.3_steel_20201217_134523.txt',
       r'D:\\xy-repo\\data\\20201217\\192.168.101.10\\T04\\T04_load_109_6000_2000_0.3_steel_20201217_134528.txt',
       r'D:\\xy-repo\\data\\20201217\\192.168.101.10\\T04\\T04_load_10_6000_2000_0.3_steel_20201217_111436.txt',
       r'D:\\xy-repo\\data\\20201217\\192.168.101.10\\T04\\T04_load_110_6000_2000_0.3_steel_20201217_134534.txt',
       r'D:\\xy-repo\\data\\20201217\\192.168.101.10\\T04\\T04_load_111_6000_2000_0.3_steel_20201217_134539.txt', ]
# t06 = [r"D:\xy-repo\data\20201218\192.168.101.10\T06\T06_load_137_6000_2000_0.5`_steel_20201218_111652.txt"]
# result = get_basic_infos(t04)
# print(result)
