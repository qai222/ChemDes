import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv('output/df_windows.csv')

ymin = min(df["fom"].values - df["fom"].values)
ymax = max(df["fom"].values + df["fom"].values)
ymin = ymin - (ymax - ymin) * 0.05
ymax = ymax + (ymax - ymin) * 0.05

xmin = min(df["amount"].values)
xmax = max(df["amount"].values)
xmin = xmin - (xmax - xmin) * 0.05
xmax = xmax + (xmax - xmin) * 0.05

ligands = sorted(set(df["ligand"].tolist()))

app = Dash(__name__)
app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                ligands,
                ligands[0],
                id='ligand_iupac_name'
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

    ]),

    dcc.Graph(id='indicator-graphic'),

])


@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('ligand_iupac_name', 'value'),
)
def update_graph(ligand_iupac_name):
    dff = df[df["ligand"] == ligand_iupac_name]

    x = dff["amount"]
    y = dff["fom"]
    colors = dff["window"]

    fig = px.scatter(x=x, y=y, color=colors)

    # fig.add_trace(go.Scatter(x=x, y=y, name="experimental", mode="markers"))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title="Ligand Amount (uL * M)", )

    fig.update_yaxes(title="Figure of Merit (a.u.)", )

    fig.update_layout(yaxis_range=[ymin, ymax], xaxis_range=[xmin, xmax])
    return fig


if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8126)
