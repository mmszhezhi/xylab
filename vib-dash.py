import plotly.graph_objects as go # or plotly.express as px
import numpy as np
N = 50
fig = go.Figure(data=[go.Mesh3d(x=(30*np.random.randn(N)),
                   y=(25*np.random.randn(N)),
                   z=(30*np.random.randn(N)),
                   opacity=0.5,)])
fig.update_layout(scene=dict(xaxis_showspikes=False,
                             yaxis_showspikes=False))

# fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
# fig.add_trace( ... )
# fig.update_layout( ... )

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])



app.run_server(debug=True, use_reloader=False)