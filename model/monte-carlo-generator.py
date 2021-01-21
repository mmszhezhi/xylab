import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os, json

cutter = {
    "T01": {
        "phase1": {
            "start": (121, 10),
            "length": (200, 100),
            "lift": (54, 30),
            "var": (4.17, 0.3),
            "ascent_var": (8, 4),
            "mean": (147, 70),
            "var_raw": (16.8, 3)
        },
        "phase2": {
            "start": None,
            "length": (30, 10),
            "lift": (350, 50),
            "var": (0, 20),
            "mean": (283, 6),
            "var_raw": (64.6, 8),
            "ascent_var": (40, 4),
        }


    },
    "T04": {
        "phase1": {
            "start": (140, 10),
            "length": (217, 10),
            "lift": (54, 30),
            "var": (2.17, 0.01),
            "ascent_var": (8, 4),
            "mean": (147, 20),
            "var_raw": (16.8, 3)
        },
        # "phase2": {
        #     "start": None,
        #     "length": (30, 2),
        #     "lift": (350, 50),
        #     "var": (2, 2),
        #     "mean": (210, 32),
        #     "var_raw": (2.6, 2),
        #     "ascent_var": (4, 4),
        # },
        "phase3": {
                    "start": None,
                    "length": (120, 5),
                    "lift": (350, 50),
                    "var": (0, 6),
                    "mean": (250, 28),
                    "var_raw": (5.6, 8),
                    "ascent_var": (4, 4),
                    "linspace":True
                },
        "phase4": {
                    "start": None,
                    "length": (25, 2),
                    "lift": (350, 50),
                    "var": (0, 40),
                    "mean": (413, 40),
                    "var_raw": (10.6, 8),
                    "ascent_var": (40, 4),
                }

    }
}


def gen_config():
    config = {}
    for k, v in cutter.items():
        c = {}
        for pno, p in v.items():
            ret = {}
            ret['start'] = np.random.normal(*p.get("start") if p.get("start") else next_start)
            ret['length'] = math.ceil(np.random.normal(*p["length"]))
            ret['lift'] = np.random.normal(*p["lift"])
            ret["mean"] = np.random.normal(*p["mean"])
            ret["ascent_var"] = np.linspace(1, 100, ret["length"]) / 100 * np.random.normal(*p["ascent_var"])
            ret["var_raw"] = np.random.normal(*p["var_raw"])
            ret['var'] = np.random.normal(*p["var"], size=ret['length'])
            ret['linspace'] = p.get("linspace")
            next_start = (ret["start"] + ret['lift'], 1)
            c.update({pno: ret})
        config.update({k: c})
    return config


def gen_data(n):
    ret = []
    for i in range(n):
        config = gen_config()

        for k, v in config.items():
            t = []
            for pno, p in v.items():
                t.append(np.linspace(p["start"], p["start"] + p['lift'], p['length']) + p["var"] + p["ascent_var"])
            ret.append(np.concatenate(t))
    return ret


def gen_data2(n, dst):
    ret = []
    j = 0
    for i in range(n):
        try:
            config = gen_config()
            for k, v in config.items():
                if k == "T01":
                    continue
                t = []
                for pno, p in v.items():
                    dt = np.random.normal(p["mean"], p["var_raw"], p["length"])

                    t.append(np.sort(dt) + p["var"] + p["ascent_var"])
                    # t.append(np.linspace(p["start"], p["start"] + p['lift'], p['length']) + p["var"])
                t[1] = list(filter(lambda x: x > max(t[0]), t[1]))
                max0, min1 = max(t[0]), min(t[1])
                shift = min1 - max0

                if shift > 15:
                    t[1] = t[1] - (shift - 15)
                data = np.concatenate(t)[10:]
                if data.shape[0] < 150 or (len(t[1]) > 50) or len(t[0]) / len(t[1]) > 4.5 or t[0][0] >142 or len(t[0]) / len(t[1]) < 2.8:
                    continue
                print(f"{i} of {n} valid {j}")
                j+=1
                wp = {"load": list(data)}
                with open(os.path.join(dst, f"mc_load_{i}.txt"), 'w') as f:
                    json.dump(wp, f)
                plt.scatter(list(range(data.shape[0])), data)
                plt.savefig(os.path.join(dst, "images", f"mc_load_{i}.png"))
                plt.clf()
        except Exception as e:
            print(repr(e))
            # ret.append(np.concatenate(t)[10:])

def max_min_shift(t1,t2,n):
    shift12 = min(t2) - max(t1)
    if shift12 > n:
        t2 = np.array(t2) - shift12
    return t2

def gen_data3(n, dst):
    ret = []
    j = 1
    for i in range(n):
        try:
            config = gen_config()
            for k, v in config.items():
                if k == "T01":
                    continue
                t = []
                for pno, p in v.items():
                    if p.get("linspace"):
                        dt = np.linspace(p['mean'],p['mean']+5,p["length"])
                        dt = dt + p["var"]
                    else:
                        dt = np.random.normal(p["mean"], abs(p["var_raw"]), p["length"])

                    t.append(np.sort(dt) + p["var"] + p["ascent_var"])
                    # t.append(np.linspace(p["start"], p["start"] + p['lift'], p['length']) + p["var"])
                t[1] = list(filter(lambda x: x > max(t[0]), t[1]))
                t[2] = list(filter(lambda x: x > max(t[1]), t[2]))
                if max(t[1]) > min(t[2]):
                    print("ff")
                t[0] = t[0][14:]
                t[1] = max_min_shift(t[0],t[1],8)
                t[2] = max_min_shift(t[1],t[2],9)


                if len(t[2]) < 11 or len(t[1]) < 80 or t[0][0] < 115 or max(t[2]) < 500:
                    continue
                print(f"{i} of {n} valid {j}")
                j+=1
                data = np.concatenate(t)[10:]
                wp = {"load": list(data)}
                with open(os.path.join(dst, f"mc_load_{i}.txt"), 'w') as f:
                    json.dump(wp, f)
                # plt.scatter(list(range(data.shape[0])), data)
                plt.figure(figsize=(10,5))
                plt.scatter(list(range(len(t[0]))),t[0],alpha=0.5)
                plt.scatter(list(range(len(t[0]) ,len(t[0]) + len(t[1]))), t[1],alpha=0.5)
                plt.scatter(list(range(len(t[0]) + len(t[1]),len(t[0]) + len(t[1]) + len(t[2]))), t[2],alpha=0.5)
                # plt.scatter(list(range(len(t[0]) + len(t[1]) + len(t[2]),len(t[0]) + len(t[1])+len(t[3]) + len(t[2]))), t[3])

                plt.savefig(os.path.join(dst, "images", f"mc_load_{i}.png"))

                plt.clf()
        except Exception as e:
            print(repr(e))
            # ret.append(np.concatenate(t)[10:])


if __name__ == '__main__':
    dst = r"C:\repo2021\xyrepo\data\mcT07-4"
    os.makedirs(dst+"\images",exist_ok=True)
    data = gen_data3(10000, dst)
    # plt.scatter(list(range(data[0].shape[0])), data[0])
    # plt.show()
    # data = np.random.normal()
