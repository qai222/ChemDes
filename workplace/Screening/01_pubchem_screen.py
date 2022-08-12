"""
download csv from
https://www.ncbi.nlm.nih.gov/pccompound?term=(%22has%20src%20vendor%22%5BFilter%5D)%20AND%20%22Sigma-Aldrich%22%5BSourceName%5D
"""
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles, Descriptors
from tqdm import tqdm

from lsal.utils import FilePath, remove_stereo, parse_formula, get_basename, write_smi


def screen_pubchem(
        pubchem_csv: FilePath,
        allowed_elements: set[str],
        mw_max: float = 400,
):
    df = pd.read_csv(pubchem_csv)[["isosmiles", "inchi", "iupacname", "mf", "cid"]]
    df.columns = ["smiles", "inchi", "iupacname", "formula", "cid"]
    good_tuples = []
    for t in tqdm(df.itertuples(index=False, name=None)):
        smiles, inchi, iupacname, formula, cid = t
        # - more than one component
        if "." in smiles:
            continue
        # - apparent charge in formula
        if "+" in formula or "-" in formula:
            continue
        # - invalid formula
        try:
            fdict = parse_formula(formula)
        except ValueError:
            continue
        # - carbon should be there
        if "C" not in fdict.keys():
            continue
        # - not a subset of allowed elements
        if not set(fdict.keys()).issubset(allowed_elements):
            continue
        smiles = remove_stereo(smiles)
        # - invalid smiles
        try:
            m = MolFromSmiles(smiles)
        except ValueError:
            continue
        # - mw larger than 400
        mw = Descriptors.ExactMolWt(m)
        if mw > mw_max:
            continue
        smiles = MolToSmiles(m)
        good_tuples.append((smiles, inchi, iupacname, formula, cid))
    df_screened = pd.DataFrame(good_tuples, columns=df.columns)
    return df_screened


if __name__ == '__main__':
    AllowedElements = {"C", "H", "O", "N", "P", "S", "Br", "F", }
    screened_df = screen_pubchem("results/PubChem_compound.csv", allowed_elements=AllowedElements, mw_max=400)
    screened_df.to_csv(get_basename(__file__) + ".csv", index=False)
    write_smi(screened_df["smiles"], get_basename(__file__) + ".smi", )
