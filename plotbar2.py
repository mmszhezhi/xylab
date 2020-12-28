import plotly as py
import plotly.graph_objs as go

if __name__ == '__main__':
    year = [8, 9, 10]

    trace_1 = go.Bar(
        x=[9],
        y=[1, 2, 3,67,545,666],
        name="甲"
    )

    trace_2 = go.Bar(
        x=[10],
        y=[4, 5, 6],
        name="乙"
    )

    trace_3 = go.Bar(
        x=[8],
        y=[7, 8, 9,12,23,56,41],

        name="丙"
    )

    trace = [trace_1, trace_2, trace_3]

    layout = go.Layout(
        title='stack堆叠对比',
        barmode='stack'
    )
    figure = go.Figure(data=trace, layout=layout)
    py.offline.plot(figure, filename='fig.html')