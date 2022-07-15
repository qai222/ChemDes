import pandas as pd

from lsal.tasks.screen import delta_plot, domain_range, get_smi2record
from lsal.utils import read_smi, get_basename

if __name__ == '__main__':
    # load pool smis
    mordred_df = pd.read_csv("results/03_descriptor_mordred.csv")
    cxcalc_df = pd.read_csv("results/02_descriptor_cxcalc.csv")
    cxcalc_smis = read_smi("results/02_descriptor_cxcalc.smi")
    assert len(mordred_df) == len(cxcalc_df) == len(cxcalc_smis)
    pool_smis = cxcalc_smis

    # combine cxcalc and mordred
    cm_df = pd.concat([cxcalc_df, mordred_df], axis=1)
    available_features = cm_df.columns.tolist()
    print("available features: {}".format(available_features))
    smi2record = get_smi2record(pool_smis, cm_df, None)

    # delta plot
    lim = domain_range("../MolDescriptors/ligand_descriptors_2022_06_16.csv", available_features)
    delta_plot(
        pool_smis, lim, cm_df,
        "results/" + get_basename(__file__) + ".png",
        available_features,
        wdir="results/{}".format(get_basename(__file__))
    )
