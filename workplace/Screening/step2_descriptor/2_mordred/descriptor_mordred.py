from lsal.tasks.descal import calculate_mordred
from lsal.utils import read_smi

if __name__ == '__main__':
    smis = read_smi("../1_cxcalc/descriptor_cxcalc.smi")
    mordred_df = calculate_mordred(smis)
    mordred_df.to_csv("descriptor_mordred.csv", index=False)
