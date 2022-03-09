from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolToInchi, MolFromSmiles
from rdkit.Chem.inchi import MolFromInchi


def inchi2smiles(inchi: str) -> str:
    return MolToSmiles(MolFromInchi(inchi))


def smiles2inchi(smi: str) -> str:
    return MolToInchi(MolFromSmiles(smi))


def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol
