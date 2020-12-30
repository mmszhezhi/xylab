from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=1, cols=2)


data ="97.91 116.36 118.22 118.71 119.39 121.14 121.34 121.59 122.75 124.7, 124.7  125.83 126.31 126.09 126.93 127.7  123.59 125.7  126.48 126.63, 125.83 127.22 126.07 126.72 126.5  127.36 128.07 128.65 128.09 128., 128.11 126.46 127.96 128.02 127.27 127.67 127.55 125.6  129.69 114.24, 135.56 136.53 136.18 136.52 136.46 136.4  136.3  136.89 136.53 136.65, 136.54 136.5  135.8  135.94 138.41 137.76 138.07 137.87 138.23 138.67, 138.42 137.89 137.74 137.49 137.34 137.54 138.26 137.36 138.47 138.55, 138.07 138.2  138.28 138.73 138.93 139.58 139.79 140.37 140.04 120.93, 139.64 141.06 140.8  141.17 141.68 141.39 141.39 142.26 141.76 140.66, 141.5  141.38 141.62 141.52 142.36 141.91 142.17 142.   142.75 143.28".split(" ")
fig.add_trace(go.Scatter(y=data, mode="markers"), row=1, col=1)
fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)

fig.show()