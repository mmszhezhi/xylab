import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import add

width = 0.3

i = 0
df = pd.read_csv("data-base-5.csv",index_col=0)
df.drop(columns=["index"], inplace=True)
# df = pd.DataFrame({"kiko":[2,3,10],"herman":[3,4,5]})
ind = df.columns
sums = [0]*df.shape[1]
for row in df.iterrows():
    i += 1
    print(i, sums)
    p2 = plt.bar(ind, row[1].values, width, bottom=sums)
    sums = list(map(add, sums, row[1]))
    # break
for x,y in zip(ind,sums):
    plt.text(x,y+10,str(y), ha='center',va='bottom')
plt.title("total power")
plt.savefig("totalpower5.png")
plt.show()
