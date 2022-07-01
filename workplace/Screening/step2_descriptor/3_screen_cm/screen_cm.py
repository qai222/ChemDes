import pandas as pd

from lsal.tasks.screen import delta_plot, domain_range, delta_feature_screen, get_smi2record
from lsal.utils import write_smis, read_smi

if __name__ == '__main__':
    # load sigma smis from pubchem
    sigma_smis = read_smi("../1_cxcalc/descriptor_cxcalc.smi")

    # combine cxcalc and mordred
    cxcalc_df = pd.read_csv("../1_cxcalc/descriptor_cxcalc.csv")
    mordred_df = pd.read_csv("../2_mordred/descriptor_mordred.csv")
    assert cxcalc_df.shape[0] == mordred_df.shape[0] == len(sigma_smis)
    cm_df = pd.concat([cxcalc_df, mordred_df], axis=1)
    available_features = cm_df.columns.tolist()
    print("available features: {}".format(available_features))
    smi2record = get_smi2record(sigma_smis, cm_df, None)

    # delta plot, output delta 0 smi
    lim = domain_range("../../../MolDescriptors/ligand_descriptors_2022_06_16.csv", available_features)
    delta_plot(sigma_smis, lim, cm_df, "screen_cm.png", available_features)
    _, screened_smis, _ = delta_feature_screen(1e-5, lim, sigma_smis, cm_df, available_features)
    print("screening produce: {}/{}".format(len(screened_smis), len(sigma_smis)))
    write_smis(screened_smis, "screen_cm.smi")
    pd.DataFrame.from_records([smi2record[s] for s in screened_smis]).to_csv("screen_cm.csv", index=False)
