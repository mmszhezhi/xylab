import numpy as np
import pandas as pd
from filereader import search, search_total_raw, get_filted_data, get_break_points,concatnate_vib,concatnate_load
import os, glob
import matplotlib.pyplot as plt
from scipy.signal import stft
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html


search_list = ("T05")
# search_list = ("T07","T08","T09","T10")

# search_list = ("T02",)
search_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
tool_path = search_total_raw(search_dir, search_list,index=(70,75))
tool_path_list = list(tool_path.values())[0]
load_data = concatnate_load(list(tool_path.values())[0])
vib_data = concatnate_vib(tool_path_list)
vdata = np.array(vib_data) -  np.mean(vib_data)
nper = np.argmax(np.where(vdata>50,1,0))

fig,ax = plt.subplots(2,1)

freq,time,sresult = stft(vib_data,5000,nperseg=nper*2)
sresult = np.abs(sresult)
sresult = np.clip(sresult,0,100)
fig = make_subplots(rows=3, cols=1,
                    specs=[[{'is_3d': True}],[{'is_3d': False}],[{'is_3d': False}]],
                    # subplot_titles=sum([[f"{x} time-frequency"] for x in cuttool],[]),

                    )



# freq_y = np.linspace(0, f(freq_0), freq_0)

fig.add_trace(go.Surface(z=sresult, x=time, y=freq,cmax=300000,cmin=-10),1,1)
# fig.add_trace(go.Bar(y=[2, 1, 3]), row=i, col=2)
# data = "97.91 116.36 118.22 118.71 119.39 121.14 121.34 121.59 122.75 124.7, 124.7  125.83 126.31 126.09 126.93 127.7  123.59 125.7  126.48 126.63, 125.83 127.22 126.07 126.72 126.5  127.36 128.07 128.65 128.09 128., 128.11 126.46 127.96 128.02 127.27 127.67 127.55 125.6  129.69 114.24, 135.56 136.53 136.18 136.52 136.46 136.4  136.3  136.89 136.53 136.65, 136.54 136.5  135.8  135.94 138.41 137.76 138.07 137.87 138.23 138.67, 138.42 137.89 137.74 137.49 137.34 137.54 138.26 137.36 138.47 138.55, 138.07 138.2  138.28 138.73 138.93 139.58 139.79 140.37 140.04 120.93, 139.64 141.06 140.8  141.17 141.68 141.39 141.39 142.26 141.76 140.66, 141.5  141.38 141.62 141.52 142.36 141.91 142.17 142.   142.75 143.28".split(" ")
# fig.add_trace(go.Scatter(y=data, mode="markers"), row=i, col=2)
# fig["layout"][f'scene{i}']["xaxis"]["title"] = "加工次数"
# fig["layout"][f'scene{i}']["yaxis"]["title"] = "频率Hz"
# fig["layout"][f'scene{i}']["zaxis"]["title"] = "幅值"
# if col ==2 and not summary.get(tool).empty:
#     data = summary.get(tool)
#     data = data.values
#     fig.add_trace(go.Scatter(y=data,mode="markers"), i, 2)


fig.update_layout(title=f'Time-frequency', autosize=False,
                  width=2500, height=4000,
                  margin=dict(l=65, r=50, b=65, t=90))


# fig.show()
# ax[0].plot(load_data)
# ax[1].plot(vib_data)
# plt.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=False, use_reloader=False,port=8060,host="127.0.0.1")
