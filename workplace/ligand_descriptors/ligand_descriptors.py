import datetime

from chemdes.schema import load_inventory
from chemdes.tasks.descal import opera_pka, calcall


def simplify_results(results: dict, important_descriptor_names: [str]):
    simplified = dict()
    for m in results:
        simplified[m] = dict()
        r = results[m]
        for k, v in r.items():
            if k.name in important_descriptor_names:
                simplified[m][k] = v
    return simplified


if __name__ == '__main__':
    # https://mordred-descriptor.github.io/documentation/master/descriptors.html
    # https://github.com/kmansouri/OPERA
    expert_descriptors = [
        # mordred
        "SLogP", "nHBDon", "nHBAcc", "nRot", "TopoPSA", "nHeavyAtom", "fragCpx", "nC", "nO", "nN", "nP", "nS", "nRing",
        # rdkit
        "FormalCharge",
        # opera
        "pKa",
        # unknown
        "?chain length", "?number of branches"
    ]

    mols = load_inventory("../data/2022_0217_ligand_InChI_mk.xlsx", to_mols=True)
    # Molecule.write_smi(mols, "mols.smi")  # write smi file for opera
    opera_results = opera_pka(mols, "mols-smi_OPERA2.7Pred.csv")
    des_df = calcall(mols, opera_results, expert_descriptors)
    des_df.to_csv("molecular_descriptors_{}.csv".format(datetime.datetime.now().strftime("%Y_%m_%d")),
                  index=False)
