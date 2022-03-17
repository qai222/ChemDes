import logging
import os
import pathlib
import typing

from chemdes.calculator import MordredCalculator, RdkitCalculator
from chemdes.schema import Molecule, pd
from chemdes.utils import to_float


def simplify_results(results: dict, important_descriptor_names: [str]) -> dict:
    logging.warning(">>> simplify_results <<<")
    logging.warning("important descriptors are:\n{}\n".format("\n".join(important_descriptor_names)))
    simplified = dict()
    for m in results:
        simplified[m] = dict()
        r = results[m]
        for k, v in r.items():
            if k.name in important_descriptor_names:
                simplified[m][k] = v
                logging.warning("ACCEPT descriptor: {}".format(k.name))
            else:
                logging.warning("REJECT descriptor: {}".format(k.name))
    return simplified


def opera_pka(mols, opera_output: typing.Union[pathlib.Path, str] = "mols-smi_OPERA2.7Pred.csv") -> dict:
    """ only return first pKa """
    # input gen for opera
    if not os.path.isfile("mols.smi"):
        Molecule.write_smi(mols, "mols.smi")
    # parse output
    opera_df = pd.read_csv(opera_output)
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
        assert pka is not None
        opera_results[m] = {"pKa": pka}
    return opera_results


def calcall(mols: [Molecule], opera_results: dict, expert_descriptors: [str]) -> pd.DataFrame:
    # mordred calculator
    mc = MordredCalculator()
    mc.calc_all(mols)
    mordred_results = simplify_results(mc.results, expert_descriptors)

    # rdkit calculator
    rc = RdkitCalculator()
    rc.calc_all(mols)
    rdkit_results = simplify_results(rc.results, expert_descriptors)

    combine_results = []
    for m in mols:
        record = dict()
        record["InChI"] = m.inchi
        record["IUPAC Name"] = m.iupac_name
        for k, v in opera_results[m].items():
            record["OPERA-" + k] = v
        for k, v in rdkit_results[m].items():
            record["RDKIT-" + k.name] = v
        for k, v in mordred_results[m].items():
            record["MORDRED-" + k.name] = v
        combine_results.append(record)
    df = pd.DataFrame.from_records(combine_results)
    assert not df.isnull().values.any()
    return df
