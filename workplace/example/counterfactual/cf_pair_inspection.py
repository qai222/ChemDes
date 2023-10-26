from pandas._typing import FilePath
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import rdFMCS

from lsal.db.document import prepare_cfpool_docs
from lsal.db.iteration_paths import load_cps
from lsal.schema import Molecule
from lsal.utils import json_dump
from lsal.utils import json_load

_SIMILARITY_THRESHOLD = 0.6  # only molecular pairs above this will be considered
_DELTA_THRESHOLD = 0.6  # only molecular pairs above this are counterfactuals


def remove_atom_by_indices(mol, indices):
    """ return a new rdkit.Mol with selected atoms removed """
    edit_mol = Chem.EditableMol(mol)
    indices = sorted(indices, reverse=True)
    for i in indices:
        edit_mol.RemoveAtom(i)
    return edit_mol.GetMol()


def get_diff_smiles(mol1_smiles, mol2_smiles):
    """ return two SMILES denoting the results after subtracting the largest common substructure """
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)

    frags_1 = []
    for atom in mol1.GetAtoms():
        if atom.GetIdx() in match1:
            frags_1.append(atom.GetIdx())
    frags_1 = remove_atom_by_indices(mol1, frags_1)

    frags_2 = []
    for atom in mol2.GetAtoms():
        if atom.GetIdx() in match2:
            frags_2.append(atom.GetIdx())
    frags_2 = remove_atom_by_indices(mol2, frags_2)
    return Chem.MolToSmiles(frags_1), Chem.MolToSmiles(frags_2)


class DiffSmiles(BaseModel):
    smiles1: str
    smiles2: str
    residual1: str
    residual2: str
    delta: float
    similarity: float
    is_cf: bool


def inspect_pairs(
        dmat_npy: FilePath = "../../../workplace_data/OneLigand/dimred/dmat_chem.npy",
        iteraction_yaml: FilePath = "../../OneLigand/visualize_mongo/iteration_paths.yaml",
        ligands_json: FilePath = "../../MolecularInventory/ligands.json.gz",
):
    ips = load_cps(iteraction_yaml)
    ligands = json_load(ligands_json)
    ligands: list[Molecule]
    ligand_dict = {ligand.label: ligand for ligand in ligands}
    ip = ips[-1]
    docs = prepare_cfpool_docs(
        ip,
        ligands,
        dmat_npy,
        specify_directed_u_score="mu_top2%mu @ top",
        base_label_only_from_suggestions=False,
        ncfs=100
    )
    smiles_diffs = []
    for d in docs:
        smilarity = d['similarity']
        if smilarity <= _SIMILARITY_THRESHOLD:
            continue
        ref = ligand_dict[d['ligand_label_base']]
        cf = ligand_dict[d['ligand_label_cf']]
        res1, res2 = get_diff_smiles(ref.smiles, cf.smiles)
        delta = d['rank_value_delta']
        ds = DiffSmiles(
            smiles1=ref.smiles,
            smiles2=cf.smiles,
            residual1=res1,
            residual2=res2,
            delta=delta,
            similarity=smilarity,
            is_cf=delta > _DELTA_THRESHOLD,
        )
        smiles_diffs.append(ds)
    return smiles_diffs


if __name__ == '__main__':
    DIFFS = inspect_pairs()
    json_dump(DIFFS, __file__.replace(".py", ".json.gz"), gz=True)
