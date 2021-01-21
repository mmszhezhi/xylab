import numpy as np
import pandas as pd
from filereader import search, search_total_raw, get_filted_data, get_break_points,concatnate_vib,concatnate_load
import os, glob
import matplotlib.pyplot as plt
from scipy.signal import stft
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 80000
import dash
import dash_core_components as dcc
import dash_html_components as html


search_list = ("T06",)
index=(135,141)
# search_list = ("T07","T08","T09","T10")

# search_list = ("T02",)
search_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
tool_path = search_total_raw(search_dir, search_list,index=index)
result = get_filted_data(tool_path)
tool_path_list = result[search_list[0]]["paths"].to_list()
# tool_path_list = list(tool_path.values())[0]
tool_path_list = sorted(tool_path_list,key=lambda x:int(os.path.basename(x).split("_")[2]))
load_data = concatnate_load(tool_path_list)
vib_data = concatnate_vib(tool_path_list)
vdata = np.array(vib_data) -  np.mean(vib_data)
nper = np.argmax(np.where(vdata>50,1,0))
fig,ax = plt.subplots(nrows=3,figsize=(13,26))
ax[0].plot(load_data)
ax[0].set_title("load")
ax[1].plot(vdata)
ax[1].set_title("vibration-x")
ax[2].specgram(x=vdata,NFFT=1000, Fs=5000, noverlap=0,mode="psd")
plt.title(f"{search_list}-{index}-psd")
plt.savefig(f"spectrum/{'-'.join([x for x in search_list])}-{index[0]}-{index[1]}.png")
plt.show()

