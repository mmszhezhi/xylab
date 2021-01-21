import os
import numpy as np
from filereader import search_total_raw, get_basic_infos, get_filted_data, extract_index_dict
import pandas as pd
import sklearn
import math, threading
from model.solvepolynomial import SolvePolynomial
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from model.models import DataGenerator,FIFODataFrame,SkewNormalDsitribution
from sklearn.linear_model import GammaRegressor,Ridge,LinearRegression
from sklearn.preprocessing import PolynomialFeatures,PowerTransformer


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
    search_list = ("T09",)
    endpoints = []
    totals = get_total(search_list[0])


    datagen = DataGenerator(search_list=search_list, wins=1)
    datagen.init_source()
    datawin = 30
    diff = 0
    stop = 700
    degree = 1
    df = FIFODataFrame(length=datawin)
    poly = PolynomialFeatures(degree=degree)
    model = Ridge(alpha=0,max_iter=100,tol=0.01,normalize=False)
    dirname = f"{search_list[0]}-{stop}-{degree}-{datawin}"
    os.makedirs(dirname, exist_ok=True)
    snd = SkewNormalDsitribution(scale=5,skew=3)
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
            tv = df.df[:3]["var"].mean()
            rv = i["var"].values / tv

            # if rv >1.3 and update:
                # df.pop(ration=rv)

        if df.df is None or df.df['index'].iloc[-1] < 20:
            df.add(i)
            continue

        df.add(i)
        n = df.df.shape[0]
        data = df.df['mean'].values
        pc = df.df['mean'].pct_change()

        # samplew = np.linspace(1,(df.df.shape[0]+1)*50,df.df.shape[0])
        # samplew = np.multiply(snd.pdf(df.df['mean'].pct_change() + 1),np.linspace(1,n,n))
        samplew = np.multiply(snd.pdf(np.log(np.clip(pc.values * 1000, a_min=-1, a_max=None) + 1)), np.linspace(1, n, n))
        # samplew = [1] * df.df.shape[0]
        samplew[0] = 0.1
        x = poly.fit_transform(np.array(df.df['index'].values).reshape([-1,1]))
        model.fit(x,df.df['mean'],sample_weight=samplew)

        test = np.linspace(-10, 20, 20).reshape([-1,1])
        cur = int(i['index'])
        test += cur
        try:
            tested = poly.transform(test)
        except:
            continue
        ypre = model.predict(tested)
        xx = np.array(list(range(int(cur),1000,1))).reshape([-1,1])
        yy = model.predict(poly.transform(xx))
        # np.concatenate([np.array(list(range(int(cur), 1000, 1))).reshape([-1, 1]), yy], axis=1)
        result = np.concatenate([xx,yy.reshape([-1,1])],axis=1)
        t = result[result[:,1] > stop]
        # j = np.min(np.argwhere(yy>stop)) if np.argwhere(yy>stop).any() else 1000
        endpoints.append((cur,t[0][0] - cur) if t.any() else (cur,1000))
        # endpoint = round(yy[j],1) if j != 1000 else None
        total = totals.iloc[:]
        plt.scatter(total['index'], total["mean"], c='skyblue', label="raw data", alpha=0.45)
        plt.scatter(df.df["index"], df.df["mean"], c='salmon', label="window",alpha=0.5)
        # plt.scatter(np.array(df.df.index) + k - len(df.df.index), df.df["mean"], c='salmon', label="training data",alpha=0.5)
        plt.plot(test, ypre, c='deeppink', label="hypothesis curve", alpha=0.5)
        plt.title(f"{search_list[0]} epoch of {k}  step {endpoints[-1][1]} endpoint {t[0][1] if t.any() else None}")
        plt.vlines(x=cur, ymin=tm - 20, ymax=tm + 20)
        bardata = np.array(endpoints)
        plt.bar(bardata[:,0],bardata[:,1],alpha=0.5,color=['mediumpurple']*bardata.shape[1])
        plt.legend()
        print(f'batch of {k} win {df.df["index"].iloc[-1]} total {total["index"].iloc[-1]} ')
        plt.savefig(f"{dirname}/{k}.png")

        plt.clf()

