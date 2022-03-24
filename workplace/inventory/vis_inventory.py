import base64
from io import BytesIO
import os.path

import dash_bootstrap_components as dbc
from dash import Dash
from dash import html
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import MolToImage

from chemdes.schema import load_inventory


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


inventory_excel = "../data/2022_0217_ligand_InChI_mk.xlsx"
df = load_inventory(inventory_excel, to_mols=False)
df = df.dropna(axis=1, how="any")
df['LigandLabel'] = df.index
df['LigandLabel'] = df['LigandLabel'].apply('{0:0>4}'.format)

df_vis = df[["LigandLabel", "Name", "Canonical SMILES"]].copy()
smiles = df.loc[:, "Canonical SMILES"]
df_vis.loc[:, "Structure"] = [smi2imagestr(smi) for smi in smiles]
df_vis.loc[:, "IUPAC Name"] = df["IUPAC Name"]
df_vis.loc[:, "InChI"] = df["InChI"]

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Row(
    [
        html.H3(
            "Load inventory from: {}".format(os.path.basename(inventory_excel)),
        style = {'margin-right': '90px', 'margin-left': '90px'}

),
        html.Div([
            dbc.Table.from_dataframe(df_vis, striped=True, bordered=True, hover=True, responsive="sm")
        ],
            style={'margin-right': '90px', 'margin-left': '90px'})
    ]
)

if __name__ == "__main__":
    df_inv = df_vis.drop("Structure", axis=1)
    df_inv.to_csv("inventory.csv", index=False)
    app.run_server(debug=False, host="0.0.0.0", port=8123)
