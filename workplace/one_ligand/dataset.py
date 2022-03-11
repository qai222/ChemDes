import pandas as pd
import re
from chemdes import *
from io import StringIO


Solvent_Identity = "m-xylene"
Ligand_Stock_Solution_Concentration = 0.05  # M




# from the first sheet of 2022_0304_LS001_MK003
Ligand_Selected = StringIO("""Name	CAS	Molecular Formula	InChI
2,2′-Bipyridyl	366-18-7	C10H8N2	InChI=1S/C10H8N2/c1-3-7-11-9(5-1)10-6-2-4-8-12-10/h1-8H
Lauric acid (dodecanoic acid)	143-07-7	C12H24O2	InChI=1S/C12H24O2/c1-2-3-4-5-6-7-8-9-10-11-12(13)14/h2-11H2,1H3,(H,13,14)
Tributylamine	102-82-9	C12H27N	InChI=1S/C12H27N/c1-4-7-10-13(11-8-5-2)12-9-6-3/h4-12H2,1-3H3
Octylphosphonic acid	4724-48-5	C8H19O3P	InChI=1S/C8H19O3P/c1-2-3-4-5-6-7-8-12(9,10)11/h2-8H2,1H3,(H2,9,10,11)
Octadecanethiol	2885-00-9	C18H38S	InChI=1S/C18H38S/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19/h19H,2-18H2,1H3""")

df_ligand = pd.read_table(Ligand_Selected)

mols = inventory_df_to_mols(df_ligand)

dfs = pd.read_excel("../data/2022_0304_LS001_MK003.xlsx", list(range(1, 7)))

emission_df = dfs[1]


def padding_vial(v: str):
    v_list = re.findall(r"[^\W\d_]+|\d+",v)
    try:
        assert len(v_list) == 2
        alpha, numeric = v_list
        return alpha + "{0:02d}".format(int(numeric))
    except (AssertionError, ValueError) as e:
        raise ValueError("invalid vial: {}".format(v))



def read_results(ligand_df:pd.DataFrame, emission_df: pd.DataFrame):
    concentration_df = ligand_df[
        [
            "Vial Site",
            "Reagent2 (ul)",
        ]
    ]

    concentration_df = concentration_df.rename(columns={
        "Vial Site": "vial",
        "Reagent2 (ul)": "ligand",
    })
    concentration_df["ligand"] *= Ligand_Stock_Solution_Concentration
    vial_list = concentration_df["vial"].tolist()
    concentration_df["vial"] = [padding_vial(v) for v in vial_list]

    print(concentration_df)
    print(emission_df)
    # solvent_volume = df["Reagent1 (ul)"]
    # ligand_solution_volume = df["Reagent2 (ul)"]
    # nc_solution_volume = df["Reagent6 (ul)"]


ligand_to_condition_df = dict(zip(mols, [dfs[i] for i in range(2, 7)]))
for mol in ligand_to_condition_df:
    condition_df = ligand_to_condition_df[mol]
    read_results(condition_df, emission_df)
    break


