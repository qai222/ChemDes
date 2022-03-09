import os

import pandas as pd

from chemdes.calculator import MordredCalculator, RdkitCalculator
from chemdes.schema import Molecule


def simplify_results(results: dict, important_descriptor_names: [str]):
    simplified = dict()
    for m in results:
        simplified[m] = dict()
        r = results[m]
        for k, v in r.items():
            if k.name in important_descriptor_names:
                simplified[m][k] = v
    return simplified


def load_mols():
    df = pd.read_excel("data/2022_0217_ligand_InChI_mk.xlsx")

    mols = []
    for s in df["InChI"]:
        if not isinstance(s, str):
            continue
        m = Molecule.from_str(s, "i")
        mols.append(m)
    return mols


if __name__ == '__main__':

    # https://mordred-descriptor.github.io/documentation/master/descriptors.html
    # https://github.com/kmansouri/OPERA
    expert_descriptors = [
        # mordred
        "SLogP",
        "nHBDon",
        "nHBAcc",
        "nRot",
        "TopoPSA",
        "nHeavyAtom",
        "fragCpx",
        "nC",
        "nO",
        "nN",
        "nP",
        "nS",
        "nRing",

        # rdkit
        "FormalCharge",

        # opera
        "pKa"

        # unknown
        "?chain length"
        "?number of branches"

    ]

    mols = load_mols()

    # mordred calculator
    mc = MordredCalculator()
    mc.calc_all(mols)
    mordred_results = simplify_results(mc.results, expert_descriptors)

    # rdkit calculator
    rc = RdkitCalculator()
    rc.calc_all(mols)
    rdkit_results = simplify_results(rc.results, expert_descriptors)

    # opera calculator
    # input gen for opera
    if not os.path.isfile("mols.smi"):
        Molecule.write_smi(mols, "mols.smi")
    # parse output
    opera_df = pd.read_csv("mols-smi_OPERA2.7Pred.csv")
    opera_df = opera_df.dropna(axis=1, how="all")
    opera_df = opera_df.drop("MoleculeID", 1)
    opera_results = dict()
    for m, r in zip(mols, opera_df.to_dict("records")):
        opera_results[m] = r

    combine_results = []
    for m in mols:
        record = dict()
        record["inchi"] = m.inchi
        for k, v in opera_results[m].items():
            record["OPERA-" + k] = v
        for k, v in rdkit_results[m].items():
            record["RDKIT-" + k.name] = v
        for k, v in mordred_results[m].items():
            record["MORDRED-" + k.name] = v
        combine_results.append(record)
    df = pd.DataFrame.from_records(combine_results)
    df.to_csv("moldes.csv", index=False)
