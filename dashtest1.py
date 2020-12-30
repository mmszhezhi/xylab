import plotly.graph_objects as go
import pandas as pd

# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

fig = go.Figure(data=go.Surface(z=z_data, showscale=False))
fig.update_layout(
    title='Mt Bruno Elevation',
    width=800, height=800,
    margin=dict(t=30, r=0, l=20, b=10)
)

name = 'eye = (x:0., y:2.5, z:0.)'
camera = dict(
    eye=dict(x=0., y=2.5, z=0.)
)


fig.update_layout(scene_camera=camera, title=name)
fig.show()