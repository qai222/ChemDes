import base64
import os.path
from io import BytesIO

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash
from dash import html
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage


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


ligand_inventory_csv = "../Inventory/ligand_inventory.csv"
df = pd.read_csv(ligand_inventory_csv)
smiles = df.loc[:, "smiles"]
df.loc[:, "Structure"] = [smi2imagestr(smi) for smi in smiles]

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Row(
    [
        html.H3(
            "Load inventory from: {}".format(os.path.basename(ligand_inventory_csv)),
            style={'margin-right': '90px', 'margin-left': '90px'}

        ),
        html.Div([
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, responsive="sm")
        ],
            style={'margin-right': '90px', 'margin-left': '90px'})
    ]
)

if __name__ == "__main__":
    port = int("8{}".format(os.path.basename(__file__)[:3]))
    app.run_server(debug=False, host="0.0.0.0", port=port)
