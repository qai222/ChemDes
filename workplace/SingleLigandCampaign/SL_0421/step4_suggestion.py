import pandas as pd
from joblib import load, dump

from lsal.schema.material import Molecule
from lsal.tasks.suggestion import suggestion_using_tuned_model


def translate_ligand_label(df: pd.DataFrame, inventory: list[Molecule], cols: list[str]):
    for col in cols:
        ligands = [Molecule.select_from_inventory(label, inventory=inventory, field="label") for label in
                   df[col]]
        df["translated_" + col] = [l.iupac_name for l in ligands]
    return df


if __name__ == '__main__':
    data = load("output/step3.pkl")
    ligand_inventory = data["step1"]["ligand_inventory"]
    ligand_to_desrecord = data["step1"]["ligand_to_desrecord"]
    tuned_reg = data["step3"]["tuned_regressor"]
    X = data["step1"]["df_X"]
    y = data["step1"]["df_y"]
    df_X_ligands = data["step1"]["df_X_ligands"]

    # exclude insoluble ligands
    insoluble_ligand_inchis = pd.read_csv("data/0421_insoluble.csv")["InChI"]
    ligand_inventory_without_insoluble = [l for l in ligand_inventory if l.inchi not in insoluble_ligand_inchis]

    df_outcome, suggestions, known_ligands, unknown_ligands = suggestion_using_tuned_model(
        tuned_regressor=tuned_reg,
        ligand_inventory=ligand_inventory_without_insoluble,
        df_X_known=X,
        df_y_known=y,
        df_ligands_known=df_X_ligands,
        ligand_to_desrecord=ligand_to_desrecord,
        compare_using_yreal=True
    )
    data["step4"] = {
        "df_outcome": df_outcome,
        "df_suggestions": suggestions,
        "known_ligands": known_ligands,
        "unknwon_ligands": unknown_ligands,
    }
    df_outcome = translate_ligand_label(df_outcome, ligand_inventory, cols=["ligand_label"])
    suggestions = translate_ligand_label(suggestions, ligand_inventory, cols=suggestions.columns)
    df_outcome.to_csv("output/suggestion_pred.csv", index=False)
    suggestions.to_csv("output/suggestion.csv", index=False)
    Molecule.write_molecules(unknown_ligands, "output/suggestion_unknown_ligands.csv", output="csv")
    Molecule.write_molecules(known_ligands, "output/suggestion_known_ligands.csv", output="csv")
    dump(data, "output/step4.pkl")
