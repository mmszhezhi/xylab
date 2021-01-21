import numpy as np
import pandas as pd
import os,glob
import math
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
import joblib


# cuttool = ["T01","T04","T05","T06"]
# cuttool = ["T01","T04","T05","T06","T07","T09","T10"]
cuttool = ["T06","T09","T10"]
# cuttool = ["T06"]

src_dir = "index_freq_norm"
with open(f"{src_dir}/weighted_len.json","r") as f:
    weighted_len = json.load(f)

fs = 5000
redo = True
draw_contrast = True
def minl_trim(data,minl):
    ret = {}
    for k,v in data.items():
        ret[k] = v["vibration-x"][:minl]
    return ret

if draw_contrast:
    try:
        col = 2
        summary = joblib.load("summary/sumdata-dict")
    except Exception as e:
        print(f"there is no summary file! {repr(e)}")
        col = 1
fig = make_subplots(rows=len(cuttool), cols=col,
                    specs=[[{'is_3d': True},{"is_3d":False}] for _ in  range(len(cuttool))],
                    subplot_titles=sum([[f"{x} time-frequency",f"load {x}"] for x in cuttool],[]),

                    )

def convert2freq(kmax,tool):
    f= lambda x: (x*5000)/weighted_len[tool]
    return list(map(f,list(range(kmax))))

for i,tool in enumerate(cuttool):
    if redo:
        i +=1
        with open(f"{src_dir}/{tool}.json","r") as f:
            data = json.load(f)
        # data = sorted(data.items(),key=lambda item:int(item[0]))
        sortedlen = sorted(data.items(),key=lambda item:len(item[1]["vibration-x"]))
        minl = len(sortedlen[0][1]["vibration-x"])
        indexoffreq = minl_trim(data,minl)
        dff = pd.DataFrame(indexoffreq)
        dff.to_csv(f"{src_dir}/freq_{tool}.csv",index=0)
        z_data = dff
        z = z_data.values
        z = z[1:,:]
        z = np.clip(z,0,1009547)
        freq_0, t_1 = z.shape
        print(tool,z.shape)
        f = lambda x: (x * 5000) / weighted_len[tool]
        # freq = convert2freq(freq_0,tool)
        # freq_max = math.ceil(max(freq))
        freq_y = np.linspace(0, f(freq_0), freq_0)
        # freq_y = np.linspace(0, freq_max, freq_max)
        t_x  = np.linspace(0, t_1, t_1)
        fig.add_trace(go.Surface(z=z, x=t_x, y=freq_y,cmax=300000,cmin=-10),i,1)
        # fig.add_trace(go.Bar(y=[2, 1, 3]), row=i, col=2)
        # data = "97.91 116.36 118.22 118.71 119.39 121.14 121.34 121.59 122.75 124.7, 124.7  125.83 126.31 126.09 126.93 127.7  123.59 125.7  126.48 126.63, 125.83 127.22 126.07 126.72 126.5  127.36 128.07 128.65 128.09 128., 128.11 126.46 127.96 128.02 127.27 127.67 127.55 125.6  129.69 114.24, 135.56 136.53 136.18 136.52 136.46 136.4  136.3  136.89 136.53 136.65, 136.54 136.5  135.8  135.94 138.41 137.76 138.07 137.87 138.23 138.67, 138.42 137.89 137.74 137.49 137.34 137.54 138.26 137.36 138.47 138.55, 138.07 138.2  138.28 138.73 138.93 139.58 139.79 140.37 140.04 120.93, 139.64 141.06 140.8  141.17 141.68 141.39 141.39 142.26 141.76 140.66, 141.5  141.38 141.62 141.52 142.36 141.91 142.17 142.   142.75 143.28".split(" ")
        # fig.add_trace(go.Scatter(y=data, mode="markers"), row=i, col=2)
        fig["layout"][f'scene{i}']["xaxis"]["title"] = "加工次数"
        fig["layout"][f'scene{i}']["yaxis"]["title"] = "频率Hz"
        fig["layout"][f'scene{i}']["zaxis"]["title"] = "幅值"
        if col ==2 and not summary.get(tool).empty:
            data = summary.get(tool)
            data = data.values
            fig.add_trace(go.Scatter(y=data,mode="markers"), i, 2)


fig.update_layout(title=f'Time-frequency', autosize=False,
                  width=2500, height=4000,
                  margin=dict(l=65, r=50, b=65, t=90))
# fig.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=False, use_reloader=False,port=8060,host="127.0.0.1")