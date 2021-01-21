import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


for cutter in ["T03","T04","T05","T06","T01","T07","T09",'T01','T02','T08','T10','T11']:
    df = pd.read_csv(f"filted_basic_info/{cutter}.csv")
    df['ptc'] = df['mean'].pct_change()
    plt.figure(figsize=(16,9))
    plt.plot(df['index'],np.clip(df['var'],a_min=0,a_max=1000) ,c='deeppink',alpha=0.5,label='var')
    plt.scatter(df['index'],np.clip(df['mean'],a_min=0,a_max=1000),label='mean')
    plt.bar(df['index'],df['ptc']*1000,color=['red' if x >0 else 'green' for x in df['ptc']],label="percent change")
    plt.title(cutter)
    plt.legend(loc='upper left')
    plt.savefig(f"loadvar/{cutter}.png")
    plt.show()



