import os
import numpy as np
from filereader import search_total_raw, get_basic_infos, get_filted_data, extract_index_dict
import pandas as pd
import sklearn
import math, threading
from model.solvepolynomial import SolvePolynomial
import matplotlib.pyplot as plt
import json


class DataGenerator():
    def __init__(self, src=None, cutters=None, search_list=None, wins=60):
        self.search_dir = src if src else os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
        self.cutters = cutters
        self.search_list = search_list if search_list else ("T01", "T04")
        self.wins = wins
        self.cache = {}

    def init_source(self):
        self.tool_path = search_total_raw(self.search_dir, self.search_list)

    def get_cached_data(self, cutter, files):
        if self.cache.get(cutter) is not None:
            indexs = extract_index_dict(files)
            indexs_needed = set(indexs.keys()) - set(self.cache[cutter]['index'])
            files = extract_index_dict(files, indexs_needed).values()

        ret = get_filted_data({cutter: files})
        if self.cache.get(cutter) is not None and ret:
            self.cache[cutter] = pd.concat([self.cache[cutter], ret[cutter]], join="outer", ignore_index=True)
        else:
            self.cache.update(ret)
        return ret

    def batchs(self):
        for k, v in self.tool_path.items():
            datal = len(v)
            for i in range(math.ceil(datal / self.wins)):
                start = i * self.wins
                end = min((i + 1) * self.wins, datal)
                # print(k, start, end)
                t = self.get_cached_data(k, v[start:end + 1])
                yield t[k]

    def iter_toolpath(self):
        for i in datagen.batchs():
            # print(i.keys())
            pass

    def cache_all(self):
        t = threading.Thread(target=self.iter_toolpath)
        t.start()
        t.join()


class WeightedRegression():

    def model_init(self):
        pass

    def model_update(self):
        pass

    def predict(self):
        pass


class FIFODataFrame():
    def __init__(self, length, df=None, *arg, **kwargs):
        self.length = length
        self.df = df
        if df is not None:
            self.init_df(df)

    def init_df(self, df):
        # if df.shape[0] < self.length:
        #     k = self.length - df.shape[0]
        #     df = pd.concat([pd.concat([df.iloc[0]] * k, axis=1).transpose(), df], ignore_index=True)
        #     df.index = df.index +1

        self.df = df.iloc[-self.length:]
        self.df.index = self.df.index + 1

    def add(self, df):
        if self.df is not None:
            self.df = pd.concat([self.df, df], ignore_index=True).iloc[-self.length:].reset_index(drop=True)
            # self.df = pd.concat([self.df, df]).iloc[-self.length:]
        else:
            self.init_df(df)

    def pop(self, index=None):
        if index and index not in self.df.index:
            raise Exception("specified index not in df")
        if not index:
            self.df.drop(axis=1, index=self.df.index[0], inplace=True).reset_index(drop=True,inplace=True)
        else:
            self.df.drop(axis=1, index=index, inplace=True).reset_index(drop=True,inplace=True)


def get_total(file):
    data = None
    try:
        data = pd.read_csv(f"../filted_basic_info/{file}.csv")
    # with open(f"../filted_basic_info/{file}.csv") as f:
    #     data = json.load(f)
    except:
        pass

    return data


if __name__ == '__main__':
    search_list = ("T07",)
    endpoints = []
    totals = get_total(search_list[0])
    df = FIFODataFrame(length=15)

    datagen = DataGenerator(search_list=search_list, wins=1)
    datagen.init_source()
    datawin = 12
    diff = 0
    stop = 700
    degree = 1
    model = SolvePolynomial(order=degree)
    dirname = f"test-{stop}-{degree}-" + search_list[0]
    os.makedirs(dirname, exist_ok=True)

    for k, i in enumerate(datagen.batchs()):
        if k == 0 or i.empty:
            continue
        update = True
        if df.df is not None:
            tm = df.df[-2:]["mean"].mean()
            diff = i["mean"].values - tm
            if diff < 0:
                update = False
                i["mean"] = tm

                # df.pop()

        if df.df is None or df.df.shape[0] < 5:
            df.add(i)
            continue
        if update:
            df.add(i)
            cur = model.polyTrain(df.df["mean"])
        x = np.linspace(-10, 20, 20)
        ypre = list(map(model.predict, x+cur))
        j = cur
        while 1:
            endpoint = round(model.predict(j),1)
            j +=1
            if endpoint > stop or j >1000:
                endpoints.append((i['index'][0],j))
                break

        total = totals.iloc[:]
        plt.scatter(total['index'], total["mean"], c='skyblue', label="raw data", alpha=0.45)
        plt.scatter(df.df["index"], df.df["mean"], c='salmon', label="training data",alpha=0.5)
        # plt.scatter(np.array(df.df.index) + k - len(df.df.index), df.df["mean"], c='salmon', label="training data",alpha=0.5)
        plt.plot(x + k, ypre, c='deeppink', label="hypothesis curve", alpha=0.5)
        plt.title(f"{search_list[0]} epoch of {k}  step {j} endpoint {endpoint}")
        plt.vlines(x=k, ymin=tm - 20, ymax=tm + 20)
        bardata = np.array(endpoints)
        plt.bar(bardata[:,0],bardata[:,1],alpha=0.5,color=['mediumpurple']*bardata.shape[1])
        plt.legend()
        print(total["index"].iloc[-1],df.df["index"].iloc[-1])
        plt.savefig(f"{dirname}/{k}.png")

        plt.clf()

    # datagen.init_source()
    # for i in datagen.batchs():
    #     print(i)
    # f = datagen.batchs()
    # next(f)
    # print(datagen.cache)
    # datagen.cache_all()
    # print(datagen.cache)
    # print("ff")
