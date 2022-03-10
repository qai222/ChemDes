import base64
from io import BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from rdkit.Chem.Draw import MolsToGridImage

from chemdes import *

"""
https://github.com/PatWalters/interactive_plots
"""
dimred_data = json_load("dimred/dimred.json")
data_2d = dimred_data["data_2d"]
molecules = dimred_data["molecules"]
x = data_2d.T[0]
y = data_2d.T[1]

graph_component = dcc.Graph(
    id='umap',
    config={'displayModeBar': False},
    figure={
        'data': [
            go.Scattergl(
                x=x,
                y=y,
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'color': 'blue',
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name="ligand"
            )
        ],
        'layout': go.Layout(
            height=400,
            xaxis={'title': 'X'},
            yaxis={'title': 'Y'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode=False,
            dragmode='select'
        )
    }
)

image_component = html.Img(id="structure-image")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([graph_component]),
    html.Div([image_component])
])


@app.callback(
    Output('structure-image', 'src'),
    [Input('umap', 'selectedData')])
def display_selected_data(selectedData):
    max_structs = 12
    structs_per_row = 6
    empty_plot = "data:image/gif;base64,R0lGODlhAQABAAAAACwAAAAAAQABAAA="
    if selectedData:
        if len(selectedData['points']) == 0:
            return empty_plot
        match_idx = [x['pointIndex'] for x in selectedData['points']]

        smiles_list = []
        name_list = []
        mol_list = []
        for i in match_idx:
            mol = molecules[i]
            mol: Molecule
            smiles_list.append(mol.smiles)
            name_list.append(mol.iupac_name)
            mol_list.append(mol.rdmol)

        img = MolsToGridImage(mol_list[0:max_structs], molsPerRow=structs_per_row, legends=name_list)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue())
        src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
    else:
        return empty_plot
    return src_str


if __name__ == '__main__':
    import socket

    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    app.run_server(debug=False, host=IPAddr)
