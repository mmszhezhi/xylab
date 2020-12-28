import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
# z_data = pd.read_csv("bruno.csv",index_col=0)
z_data.drop(columns=z_data.columns[0],inplace=True)
z_data.to_csv("bruno.csv",index=False)
# z_data.to_csv("bruno.csv",index=False)
z = z_data.values
sh_0, sh_1 = z.shape
x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=1500, height=1500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()