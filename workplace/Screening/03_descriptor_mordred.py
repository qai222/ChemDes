from lsal.tasks.descal import calculate_mordred
from lsal.utils import read_smi, get_basename

if __name__ == '__main__':
    smis = read_smi("results/02_descriptor_cxcalc.smi")
    mordred_df = calculate_mordred(smis)
    mordred_df.to_csv(get_basename(__file__) + ".csv", index=False)
