import tqdm
from loguru import logger

from lsal.schema import load_featurized_molecules
from lsal.tasks import calculate_complexities
from lsal.utils import remove_stereo, json_dump, json_load

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

inchi_to_cas = json_load("find_cas/inchi_to_cas.json.gz", gz=True)

if __name__ == '__main__':
    logger.add(sink=f'{__file__}.log')
    init_dataset_smiles = [remove_stereo(m.smiles) for m in init_dataset]
    pubchem_dataset = [m for m in pubchem_dataset if remove_stereo(m.smiles) not in init_dataset_smiles]
    ligands = init_dataset + pubchem_dataset

    # add complexity and cas number
    # only these with cas rn are included
    for lig in tqdm.tqdm(set(ligands)):
        try:
            cas_rn = inchi_to_cas[lig.identifier]
            assert cas_rn is not None
        except (KeyError, AssertionError) as e:
            logger.warning(f"DISCARD LIGAND AS CAS RN NOT FOUND: {lig.label} {lig.smiles}")
            continue
        complexity = calculate_complexities(lig.smiles)
        lig.properties['complexity_sa_score'] = complexity['sa_score']
        lig.properties['complexity_BertzCT'] = complexity['BertzCT']
        lig.properties['cas_number'] = cas_rn

    json_dump(ligands, "ligands.json.gz", gz=True)
