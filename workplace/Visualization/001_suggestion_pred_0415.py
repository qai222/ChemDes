import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv('../SingleLigandCampaign/SL_0415/output/suggestion_pred.csv')

ymin = min(df["mean"].values - df["std"].values)
ymax = max(df["mean"].values + df["std"].values)
ymin = ymin - (ymax - ymin) * 0.05
ymax = ymax + (ymax - ymin) * 0.05

xmin = min(df["amount"].values)
xmax = max(df["amount"].values)
xmin = xmin - (xmax - xmin) * 0.05
xmax = xmax + (xmax - xmin) * 0.05

ligands = sorted(set(df["translated_ligand_label"].tolist()))
ligands_labelled = sorted(set(df.dropna(subset=["y_real"])["translated_ligand_label"].tolist()))

update_ligand_names = []
for n in df["translated_ligand_label"]:
    if n in ligands_labelled:
        update_ligand_names.append("LABELLED--" + n)
    else:
        update_ligand_names.append(n)
df["ligand"] = update_ligand_names
ligands = sorted(set(df["ligand"].tolist()))
ligands_labelled = sorted(set(df.dropna(subset=["y_real"])["ligand"].tolist()))


app = Dash(__name__)
app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                ligands,
                ligands_labelled[0],
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
    y = dff["mean"]
    ye = dff["std"]


    fig = px.scatter(x=x, y=y, error_y=ye,)

    if ligand_iupac_name in ligands_labelled:
        yreal = dff["y_real"]
        fig.add_trace(go.Scatter(x=x, y=yreal, name="experimental", mode="markers"))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title="Ligand Amount (uL * uM)",)

    fig.update_yaxes(title="Figure of Merit (a.u.)",)

    fig.update_layout(yaxis_range=[ymin, ymax], xaxis_range=[xmin, xmax])
    return fig


if __name__ == '__main__':
    import os
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=False, host="0.0.0.0", port=port)
