import os
import numpy as np
from filereader import search_total_raw, get_basic_infos, get_filted_data, extract_index_dict
import pandas as pd
import sklearn
import math, threading
from model.solvepolynomial import SolvePolynomial,SGDRidgeRegression
import matplotlib.pyplot as plt
import json
from model.models import DataGenerator,get_total,FIFODataFrame




if __name__ == '__main__':
    search_list = ("T01",)
    endpoints = []
    model = SGDRidgeRegression(n_epochs=10, regularization_strength=10,learning_rate=0.5,batch_size=None)
    totals = get_total(search_list[0])
    df = FIFODataFrame(length=15)

    datagen = DataGenerator(search_list=search_list, wins=1)
    datagen.init_source()
    datawin = 12
    diff = 0
    stop = 800
    degree = 1
    # model = SolvePolynomial(order=degree)
    dirname = f"{search_list[0]}-{stop}-{degree}"
    os.makedirs(dirname, exist_ok=True)

    for k, i in enumerate(datagen.batchs()):
        if k == 0 or i.empty:
            continue
        if k ==71:
            print('f')
        update = True
        if df.df is not None:
            tm = df.df[-2:]["mean"].mean()
            diff = i["mean"].values - tm
            if diff < 0:
                update = False
                cur +=1
                i["mean"] = tm
            tv = df.df[:3]["var"].mean()
            rv = i["var"].values / tv
            if rv >1.3 and update:
                df.pop(ration=rv)

        if df.df is None or df.df['index'].iloc[-1] < 5:
            df.add(i)
            continue
        if update or not getattr(model,'predict',None):
            df.add(i)
            # model.fit(df.df['index'],df.df["mean"].values)
            model.fit(df.df['index'].values.reshape((-1, 1)), df.df["mean"].values.reshape((-1, 1)))
            cur = df.df['index'].iloc[-1]

        x = np.linspace(-10, 20, 20).reshape((-1,1))
        ypre = model.predict(x+cur)
        j = cur
        while 1:
            endpoint = round(model.predict(j),1)
            j +=1
            if endpoint > stop or j >1000:
                endpoints.append((i['index'][0],j))
                break

        total = totals.iloc[:]
        plt.scatter(total['index'], total["mean"], c='skyblue', label="raw data", alpha=0.45)
        plt.scatter(df.df["index"], df.df["mean"], c='salmon', label="window",alpha=0.5)
        # plt.scatter(np.array(df.df.index) + k - len(df.df.index), df.df["mean"], c='salmon', label="training data",alpha=0.5)
        plt.plot(x + k, ypre, c='deeppink', label="hypothesis curve", alpha=0.5)
        plt.title(f"{search_list[0]} epoch of {k}  step {j} endpoint {endpoint}")
        plt.vlines(x=k, ymin=tm - 20, ymax=tm + 20)
        bardata = np.array(endpoints)
        plt.bar(bardata[:,0],bardata[:,1],alpha=0.5,color=['mediumpurple']*bardata.shape[1])
        plt.legend()
        print(f'batch of {k} win {df.df["index"].iloc[-1]} total {total["index"].iloc[-1]} ')
        plt.savefig(f"{dirname}/{k}.png")

        plt.clf()

