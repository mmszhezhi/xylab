import numpy as np
import pandas as pd
import os,glob
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html



cuttool = ["T01","T04","T05","T06"]
redo = True
def minl_trim(data,minl):
    ret = {}
    for k,v in data.items():
        ret[k] = v["vibration-x"][:minl]
    return ret

for too in cuttool:
    if redo:
        with open(f"index_freq/{cuttool}.json","r") as f:
            data = json.load(f)
        # data = sorted(data.items(),key=lambda item:int(item[0]))
        sortedlen = sorted(data.items(),key=lambda item:len(item[1]["vibration-x"]))
        minl = len(sortedlen[0][1]["vibration-x"])
        indexoffreq = minl_trim(data,minl)
        dff = pd.DataFrame(indexoffreq)
        dff.to_csv(f"index_freq/freq_{cuttool}.csv",index=0)



z_data = dff
z = z_data.values
z = np.clip(z,0,1009547)
freq_0, t_1 = z.shape
print(z.shape)
freq_y = np.linspace(0, freq_0, freq_0)
t_x  = np.linspace(0, t_1, t_1)
fig = go.Figure(data=[go.Surface(z=z, x=t_x, y=freq_y,cmax=300000,cmin=-10)])
fig.update_layout(title=f'{cuttool} time-frequency', autosize=False,
                  width=1600, height=900,
                  margin=dict(l=65, r=50, b=65, t=90))
# fig.show()

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=False, use_reloader=False,port=8060,host="127.0.0.1")