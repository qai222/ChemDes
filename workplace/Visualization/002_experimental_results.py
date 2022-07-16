import math
import os.path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

from lsal.campaign.loader import  Molecule, Any
from lsal.campaign.fom import FomCalculator, PropertyGetter
from lsal.utils import get_basename
from lsal.utils import json_load

reaction_collection_file = "../SingleLigandCampaign/data/collect_reactions_SL_0519.json"
reaction_collection = json_load(reaction_collection_file)

# calculate foms
fomc = FomCalculator(reaction_collection)
reaction_collection = fomc.update_foms()

PropertyNames = list(PropertyGetter.NameToSuffix.keys())
LigandData = [
    PropertyGetter.get_amount_property_data(reaction_collection, pname) for pname in PropertyNames
]
Ligands = sorted(LigandData[0].keys())

def get_dfs(ligand_to_results: dict[Molecule, dict[str, Any]]):
    dfs = []
    for ligand in Ligands:
        data = ligand_to_results[ligand]
        df = pd.DataFrame()
        # df["smiles"] = [l.smiles for l in ligands]
        # df["name"] = ["{}:{}".format(l.int_label, l.name) for l in ligands]
        df["x"] = data["amount"]
        df["y"] = data["values"]
        df["identifier"] = data["identifiers"]
        y_ref = np.array([np.mean(data["ref_values"]), ] * len(data["amount"]))
        y_ref_err = np.std(data["ref_values"])
        df["y+3*sigma"] = y_ref + 3 * y_ref_err
        df["y-3*sigma"] = y_ref - 3 * y_ref_err
        dfs.append(df)
    return dfs

DFS = [get_dfs(ld) for ld in LigandData]

app = Dash(__name__)

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                PropertyNames,
                PropertyNames[0],
                id='reaction_property'
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

    ]),

    dcc.Graph(id='graph'),
    dcc.Tooltip(id="graph-tooltip"),

])


@app.callback(
    Output('graph', 'figure'),
    Input('reaction_property', 'value'),
)
def update_graph(reaction_property: str):
    assert reaction_property in PropertyNames
    prop_id = PropertyNames.index(reaction_property)
    dfs = DFS[prop_id]

    # subplot titles
    titles = []
    for ligand in Ligands:
        title = "{}:{}".format(ligand.int_label, ligand.name)
        titles.append(title)

    # subplot layout
    ncols = 5
    nrows = (len(dfs) // ncols) + 1
    fig = make_subplots(rows=nrows + 1, cols=ncols, subplot_titles=titles)

    ys = []
    xs = []
    for i, df in enumerate(dfs):
        if i == 0:
            showlegend = True
        else:
            showlegend = False
        x = df["x"].tolist()
        y = df["y"].tolist()
        xs += x
        ys += y
        irow = i // ncols
        icol = i % ncols
        fig.append_trace(
            go.Scatter(
                x=x, y=y, name=reaction_property, mode="markers",
                text=df["identifier"],
                marker=dict(color="black"),
                hovertemplate=
                "<b>%{text}</b><br>" +
        "ligand amount: %{x:,.2f}<br>" + reaction_property+
        ": %{y:,.2f}<br>" +
                "<extra></extra>",
                showlegend=showlegend,
            ),
            row=irow + 1, col=icol + 1
        )

    fig.update_layout(
        height=300 * (nrows + 1), width=300 * ncols,
        title_text=get_basename(reaction_collection_file),
        # showlegend=True,
        legend_title_text='Experimental Data',
    )
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    for i, df in enumerate(dfs):
        if i == 0:
            showlegend = True
        else:
            showlegend = False
        irow = i // ncols
        icol = i % ncols
        yerr_u = df["y+3*sigma"].tolist()
        yerr_l = df["y-3*sigma"].tolist()
        err_x = np.linspace(minx, maxx, len(yerr_u))
        fig.add_trace(
            go.Scatter(x=err_x, y=yerr_u,
                       fill=None,
                       name="ref +/- 3*sigma",
                       mode='lines',
                       hoverinfo='skip',
                       line=dict(color='gray'),
                       showlegend=showlegend
                       ), row=irow + 1, col=icol + 1
        )
        fig.add_trace(
            go.Scatter(
                x=err_x,
                y=yerr_l,
                fill='tonexty',  # fill area between trace0 and trace1
                mode='lines',
                fillcolor="rgba(255, 255, 255, 0.5)",
                hoverinfo='skip',
                line=dict(color='gray'),
                showlegend=False,
            ), row=irow + 1, col=icol + 1
        )

    xaxis_min = math.log10(minx) - 0.1
    xaxis_max = math.log10(maxx) + 0.1
    fig.update_xaxes(type="log", range=[xaxis_min, xaxis_max])
    fig.update_yaxes(range=[miny * 0.9, maxy * 1.1])  # linear range
    return fig


if __name__ == "__main__":
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=True, host="0.0.0.0", port=port)
