import glob
import os.path

import pandas as pd


if __name__ == '__main__':

    for suggestion_csv in glob.glob("suggestion/suggestion__*.csv"):
        bn = os.path.basename(suggestion_csv)
        df_dup = pd.read_csv(f"../learning_AL0303_hasdup/suggestion/{bn}")
        df = pd.read_csv(suggestion_csv)
        df_label = set(df['ligand_label'].tolist())
        df_dup_label = set(df_dup['ligand_label'].tolist())
        print("=" * 12)
        print(bn)
        print("no dup", len(df_label))
        print("has dup", len(df_dup_label))
        print("intersection", len(df_dup_label.intersection(df_label)))
