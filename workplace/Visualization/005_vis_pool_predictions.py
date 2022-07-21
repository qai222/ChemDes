import os.path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from lsal.utils import pkl_load

Available_FOMs = [
    "fom1",
    "fom2",
    "fom3"
]
Y_DEFs = ["sa_score", "BertzCT", ]
X_DEFs = ["mu", "uci", "std", "mu-top2%", "uci-top2%", "std-top2%mu", "std-top2%uci"]
SMI_data = pkl_load("../SingleLigandCampaign/al_predict/vis_predictions/export_vis_data_ligands.pkl")


def get_df_by_fom(fom_name: str):
    df = pd.read_csv(f"../SingleLigandCampaign/al_predict/vis_predictions/predictions_{fom_name}.csv")
    smis = df["smiles"].tolist()
    for ydef in Y_DEFs:
        values = [SMI_data[smi][ydef] for smi in smis]
        df[ydef] = values
    df["pool"] = [1, ] * len(df)

    df_learned = pd.read_csv(f"../SingleLigandCampaign/al_predict/vis_predictions/predictions_{fom_name}_learned.csv")
    smis = df_learned["smiles"].tolist()
    for ydef in Y_DEFs:
        values = [SMI_data[smi][ydef] for smi in smis]
        df_learned[ydef] = values
    df_learned["pool"] = [0, ] * len(df_learned)
    final_df = pd.concat([df, df_learned], axis=0)
    return final_df


DFs = {fn: get_df_by_fom(fn) for fn in Available_FOMs}
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Div([
        html.Div([
            html.P("FOM:"),
            dcc.Dropdown(
                Available_FOMs,
                Available_FOMs[0],
                id='fom_type'
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            html.P("y axis:"),
            dcc.Dropdown(
                Y_DEFs,
                Y_DEFs[0],
                id='ydef'
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            html.P("x axis:"),
            dcc.Dropdown(
                X_DEFs,
                X_DEFs[0],
                id='xdef'
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),

    ]),
    dcc.Graph(id='graph'),

    html.Div(children=[],
             style={'margin-right': '90px', 'margin-left': '90px'},
             id="ligand_table"
             )

])


@app.callback(
    Output('graph', 'figure'),
    [
        Input('fom_type', 'value'),
        Input('xdef', 'value'),
        Input('ydef', 'value'),
    ]
)
def update_graph(fom_type, xdef, ydef):
    df = DFs[fom_type]
    df_pool = df.query("pool == 1")
    df_learned = df.query("pool == 0")
    fig = go.Figure(
        data=go.Scattergl(
            name="pool",
            x=df_pool[xdef], y=df_pool[ydef], mode='markers',
            customdata=df_pool["smiles"],
            marker=dict(
                color='rgba(135, 206, 250, 0.5)',
                size=5,
            )
        )
    )
    fig.add_trace(
        go.Scattergl(
            name="real",
            x=df_learned[xdef], y=df_learned[ydef], mode="markers",
            customdata=df_learned["smiles"],
            marker=dict(
                color='rgba(255, 0, 0, 0.9)',
                size=5,
            )
        )
    )
    fig.update_layout(
        dict(height=400,
             xaxis={'title': xdef},
             yaxis={'title': ydef},
             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             legend={'x': 1, 'y': 1},
             hovermode=False,
             dragmode='select'
             )
    )
    return fig


@app.callback(
    Output('ligand_table', 'children'),
    [
        Input('fom_type', 'value'),
        Input('xdef', 'value'),
        Input('ydef', 'value'),
        Input('graph', 'selectedData')
    ]
)
def update_table(fom_type, xdef, ydef, selectedData):
    if selectedData:
        df = DFs[fom_type]
        df_learned = df.query("pool == 0")
        learned_smiles = df_learned["smiles"].tolist()
        points = selectedData["points"]
        selected_records = []
        for point in points:
            if point["customdata"] in learned_smiles:
                learned = "YES"
            else:
                learned = "NO"
            record = {
                "smiles": point["customdata"],
                "structure": html.Img(src=SMI_data[point["customdata"]]["img"]),
                xdef: "{:#.4g}".format(point["x"]),
                ydef: "{:#.4g}".format(point["y"]),
                "fom": fom_type,
                "learned": learned,
            }
            selected_records.append(record)
        selected_records = sorted(selected_records, key=lambda x: x[xdef], reverse=True)
        df_table = pd.DataFrame.from_records(selected_records)
        return [dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True, responsive="sm")]
    else:
        return []


if __name__ == "__main__":
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=False, host="0.0.0.0", port=port)
