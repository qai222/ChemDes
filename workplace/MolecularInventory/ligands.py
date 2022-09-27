import tqdm

from lsal.schema import load_featurized_molecules
from lsal.tasks import calculate_complexities
from lsal.utils import remove_stereo, json_dump

"""
combine both datasets, remove duplicates
"""

init_dataset = load_featurized_molecules(
    "initial_dataset/init_inv.csv",
    "initial_dataset/init_des.csv",
    'LIGAND',
    col_to_mol_kw={
        'label': 'label',
        'identifier': 'identifier',
        'smiles': 'smiles',
        'name': 'name',
    },
)
pubchem_dataset = load_featurized_molecules(
    "pubchem_screen/screened_inv.csv",
    "pubchem_screen/screened_des.csv",
    'POOL',
    col_to_mol_kw={
        'label': 'label',
        'identifier': 'identifier',
        'smiles': 'smiles',
    },
)
if __name__ == '__main__':
    init_dataset_smiles = [remove_stereo(m.smiles) for m in init_dataset]
    pubchem_dataset = [m for m in pubchem_dataset if remove_stereo(m.smiles) not in init_dataset_smiles]
    ligands = init_dataset + pubchem_dataset

    # add complexity
    for lig in tqdm.tqdm(ligands):
        complexity = calculate_complexities(lig.smiles)
        lig.properties['complexity_sa_score'] = complexity['sa_score']
        lig.properties['complexity_BertzCT'] = complexity['BertzCT']

    json_dump(ligands, "ligands.json.gz", gz=True)
