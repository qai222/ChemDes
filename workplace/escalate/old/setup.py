import pprint
import escalateclient
import importlib
import pandas as pd



if __name__ == '__main__':

    importlib.reload(escalateclient)
    server_url = 'http://localhost:8000'
    username = "qai1"
    password = "eBYvN4ELjrLXMDt"
    client = escalateclient.ESCALATEClient(server_url, username, password)

    # get our organization, we should have a lab with `shourt_name` == `MNL`
    mnl_lab = client.get(endpoint="organization", data={"short_name": "MNL"})  # note this always returns a list
    assert len(mnl_lab) == 1
    mnl_lab = mnl_lab[0]

# we use 1. canonical smiles 2. inchi 3. name 4. iupac name 5. label to identify ligands, they are defined as `material-identifier-def` in API
# see https://github.com/darkreactions/ESCALATE/blob/master/escalate/TECHNICAL.md for more definitions
mids = []
for name in ["SMILES", "InChI", "Name", "IUPAC Name", "Label"]:
    mid = client.get_or_create(
        endpoint="material-identifier-def",
        data={"description": name}
    )
    mids += mid
pprint.pprint(mids)
assert len(mids) == 5
mid_smiles, mid_inchi, mid_name, mid_iupac, mid_label = mids