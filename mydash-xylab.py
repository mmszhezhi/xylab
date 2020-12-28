import numpy as np
import pandas as pd
import os,glob
import json
import matplotlib.pyplot as plt


with open("trimed_vib_data/T05.json","r") as f:
    data = json.load(f)


import math
indexf = {}
mean_len = 14638
for i,item in data.items():
    x = np.array(item["vibration-x"])  - 39362.010317
    freq = np.fft.fft(x)
    h = math.ceil(len(freq) * 0.5)
    x0f = abs(freq[:h])
    indexf[i] = x0f[:7000]
dff = pd.DataFrame(indexf)
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Read data from a csv
# z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
# z_data = pd.read_csv("bruno.csv",index_col=0)
z_data = dff
# z_data.drop(columns=z_data.columns[0],inplace=True)
# z_data.to_csv("bruno.csv",index=False)
# z_data.to_csv("bruno.csv",index=False)
z = z_data.values - 200
freq_0, t_1 = z.shape
print(z.shape)
freq_y = np.linspace(0, 1, freq_0)
t_x  = np.linspace(0, 1, t_1)
fig = go.Figure(data=[go.Surface(z=z, x=t_x, y=freq_y)])
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=900, height=900,
                  margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=False, use_reloader=False,port=8060,host="127.0.0.1")