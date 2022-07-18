import base64
import os.path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage

from lsal.campaign.fom import FomCalculator, PropertyGetter
from lsal.campaign.loader import Molecule, Any
from lsal.utils import json_load
from lsal.utils import truncate_distribution

reaction_collection_file = "../SingleLigandCampaign/data/collect_reactions_SL_0519.json"
reaction_collection = json_load(reaction_collection_file)

# calculate foms
fomc = FomCalculator(reaction_collection)
reaction_collection = fomc.update_foms()

FomNames = [n for n in PropertyGetter.NameToSuffix.keys() if n.startswith("fom")]

LigandData = {pname:
                  PropertyGetter.get_amount_property_data(reaction_collection, pname) for pname in FomNames
              }
Ligands = sorted([lc[0] for lc in reaction_collection.unique_lcombs])

point_methods = ["best", "top2%", "top5%", "mean"]


def smi2imagestr(smi: str, to_html=True):
    m = MolFromSmiles(smi)
    img = MolToImage(m)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue())
    src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
    if to_html:
        return (html.Img(src=src_str))
    else:
        return src_str


def get_points(ligand_to_results: dict[Molecule, dict[str, Any]], method="best"):
    # x -> descriptor of fom values of all conc values
    # y -> ref
    points = []
    for ligand in Ligands:
        data = ligand_to_results[ligand]
        values = data["values"]
        y = np.mean(data["ref_values"])
        y_err = 3 * np.std(data["ref_values"])
        if method == "best":
            x = max(values)
        elif method == "top2%":
            x = np.mean(truncate_distribution(values, "top", 0.02))
        elif method == "top5%":
            x = np.mean(truncate_distribution(values, "top", 0.05))
        elif method == "mean":
            x = np.mean(values)
        else:
            raise ValueError
        pt = {
            "x": x,
            "y": y,
            "yerr": y_err,
            "smiles": ligand.smiles,
            "name": "{}:".format(ligand.int_label) + ligand.name,
        }
        points.append(pt)
    return points


app = Dash(__name__)

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                FomNames,
                FomNames[0],
                id='reaction_property'
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                point_methods,
                point_methods[0],
                id='method'
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

    ]),

    dcc.Graph(id='graph', clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),

])


@app.callback(
    Output('graph', 'figure'),
    Input('reaction_property', 'value'),
    Input('method', 'value'),
)
def update_graph(fom_name: str, method: str):
    points = get_points(LigandData[fom_name], method)
    x = [p["x"] for p in points]
    y = [p["y"] for p in points]
    y = [0 if pd.isna(yy) else yy for yy in y]
    y_err = [p["yerr"] for p in points]
    fig = go.Figure(
        data=go.Scatter(
            x=x, y=y, mode='markers',
            error_y=dict(
                type='data',  # value of error bar given in data coordinates
                array=y_err,
                symmetric=True,
                visible=True)
        )
    )
    return fig


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    ligand = Ligands[num]
    ligand: Molecule
    img_src = smi2imagestr(ligand.smiles, False)

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H4(f"{ligand.label}", style={"color": "darkblue"}),
            html.P(f"{ligand.name}"),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=False, host="0.0.0.0", port=port)
