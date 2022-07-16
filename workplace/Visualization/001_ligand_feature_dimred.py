import base64
import os.path
from io import BytesIO

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from lsal.schema import Molecule
from lsal.utils import FilePath, json_load

"""
visualize dimensionality reduction from ligand features
"""


# load data
def load_data(datafile: FilePath):
    data = json_load(datafile)
    data_2d = data["data_2d"]
    molecules = data["molecules"]
    return data_2d, molecules


# get dataframe
def get_vis_df(data_2d, molecules: list[Molecule]):
    records = []
    for (x, y), molecule in zip(data_2d, molecules):
        r = {
            "x": x, "y": y, "smiles": molecule.smiles, "label": str(molecule.int_label), "name": molecule.name
        }
        records.append(r)
    return pd.DataFrame.from_records(records)


DATA_2D, MOLECULES = load_data("../Inventory/dimred.json")
DF = get_vis_df(DATA_2D, MOLECULES)

# dash graphs
graph_component = dcc.Graph(
    id='mol-scatter',
    config={'displayModeBar': False},
    figure={
        'data': [
            go.Scattergl(
                x=DF.x,
                y=DF.y,
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'color': 'black',
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name="Ligand"
            ),
        ],
        'layout': go.Layout(
            height=400,
            xaxis={'title': 'Axis 1'},
            yaxis={'title': 'Axis 2'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode=False,
            dragmode='select'
        )
    }
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

image_component = html.Img(id="structure-image")
app.layout = html.Div([
    html.Div([graph_component]),
    html.Div([image_component])
])


@app.callback(
    Output('structure-image', 'src'),
    [Input('mol-scatter', 'selectedData')])
def display_selected_data(selectedData):
    max_structs = 12
    structs_per_row = 6
    empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
    if selectedData:
        if len(selectedData['points']) == 0:
            return empty_plot
        points = selectedData["points"]
        match_idx = []
        for point in points:
            pid = point["pointIndex"]
            match_idx.append(pid)
        match_df = DF.iloc[match_idx]
        # print(match_df)
        smiles_list = list(match_df.smiles)

        name_list = [list(match_df.label)[i] + ":" + list(match_df.name)[i] for i in range(len(match_df))]
        mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
        img = MolsToGridImage(mol_list[0:max_structs], molsPerRow=structs_per_row, legends=name_list)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue())
        src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
    else:
        return empty_plot
    return src_str


if __name__ == '__main__':
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=False, host="0.0.0.0", port=port)
