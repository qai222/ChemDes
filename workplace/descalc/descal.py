import datetime

from chemdes import *


def simplify_results(results: dict, important_descriptor_names: [str]):
    simplified = dict()
    for m in results:
        simplified[m] = dict()
        r = results[m]
        for k, v in r.items():
            if k.name in important_descriptor_names:
                simplified[m][k] = v
    return simplified


def opera_pka(mols):
    """only return first pKa"""
    # input gen for opera
    if not os.path.isfile("mols.smi"):
        Molecule.write_smi(mols, "mols.smi")
    # parse output
    opera_df = pd.read_csv("mols-smi_OPERA2.7Pred.csv")
    opera_df = opera_df.dropna(axis=1, how="all")
    opera_df = opera_df.drop(labels="MoleculeID", axis=1)
    # opera_df = opera_df[[c for c in opera_df.columns if c.endswith("_pred")]]
    opera_results = dict()
    for m, r in zip(mols, opera_df.to_dict("records")):
        pka_1 = to_float(r["pKa_a_pred"])
        pka_2 = to_float(r["pKa_b_pred"])
        pka = pka_1
        if pka_1 is None:
            pka = pka_2
        assert not pka is None
        opera_results[m] = {"pKa": pka}
    return opera_results


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

    mols = load_inventory("../data/2022_0217_ligand_InChI_mk.xlsx", to_mols=True)

    # mordred calculator
    mc = MordredCalculator()
    mc.calc_all(mols)
    mordred_results = simplify_results(mc.results, expert_descriptors)

    # rdkit calculator
    rc = RdkitCalculator()
    rc.calc_all(mols)
    rdkit_results = simplify_results(rc.results, expert_descriptors)

    # opera calculator
    opera_results = opera_pka(mols)

    combine_results = []
    for m in mols:
        record = dict()
        record["InChI"] = m.inchi
        for k, v in opera_results[m].items():
            record["OPERA-" + k] = v
        for k, v in rdkit_results[m].items():
            record["RDKIT-" + k.name] = v
        for k, v in mordred_results[m].items():
            record["MORDRED-" + k.name] = v
        combine_results.append(record)
    df = pd.DataFrame.from_records(combine_results)
    assert not df.isnull().values.any()
    df.to_csv("../data/molecular_descriptors_{}.csv".format(datetime.datetime.now().strftime("%Y_%m_%d")), index=False)
