import os
import numpy as np
from filereader import search_total_raw, get_basic_infos, get_filted_data, extract_index_dict
import pandas as pd
import sklearn
import math, threading
from model.solvepolynomial import SolvePolynomial
import matplotlib.pyplot as plt
import json
from scipy import stats


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
                t = self.get_cached_data(k, v[start:end])
                yield t[k]

    def iter_toolpath(self):
        for i in self.batchs():
            # print(i.keys())
            pass

    def cache_all(self):
        t = threading.Thread(target=self.iter_toolpath)
        t.start()
        t.join()


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

    def pop(self, index=None, ration=None):
        if index and index not in self.df.index:
            raise Exception("specified index not in df")
        if not index:
            if self.df.shape[0] > 6:
                self.df.drop(axis=1, index=self.df.index[:3], inplace=True)
                self.df.reset_index(drop=True, inplace=True)
            if self.df.shape[0] > 3 and ration > 10:
                self.df.drop(axis=1, index=self.df.index[:math.ceil(self.df.shape[0] / 2)], inplace=True)
                self.df.reset_index(drop=True, inplace=True)
        else:
            self.df.drop(axis=1, index=index, inplace=True)
            self.df.reset_index(drop=True, inplace=True)


def get_total(file):
    data = None
    try:
        data = pd.read_csv(f"../filted_basic_info/{file}.csv")
    # with open(f"../filted_basic_info/{file}.csv") as f:
    #     data = json.load(f)
    except:
        pass

    return data


class SkewNormalDsitribution:
    def __init__(self, skew, scale, local=0):
        self.local = local
        self.skew = skew
        self.scale = scale

    def pdf(self, x):
        t = (x - self.local) / self.scale
        return 2.0 * self.scale * stats.norm.pdf(t) * stats.norm.cdf(self.skew * t)
