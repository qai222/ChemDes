import pandas as pd
import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import typing
from lsal.one_ligand import ReactionNcOneLigand, Molecule, ReactionCondition, ReactantSolvent, ReactantSolution, \
    reactions_to_xy, categorize_reactions
from lsal.schema import inventory_df_to_mols
from lsal.schema import load_inventory
from lsal.utils import strip_extension, json_dump
from pathlib import Path
from typing import Union
from lsal.one_ligand import ReactionNcOneLigand, Molecule, ReactionCondition, categorize_reactions
import os
from lsal.schema import inventory_df_to_mols
from lsal.schema import load_inventory, ReactantSolution, ReactantSolvent
from lsal.utils import padding_vial_label, get_folder, get_basename, FilePath

this_dir = os.path.abspath(os.path.dirname(__file__))

_ligand_inventory = load_inventory("../data/2022_0217_ligand_InChI_mk.xlsx", to_mols=True)
_SolventIdentity = "m-xylene"
_LigandStockSolutionConcentrationUnit = "M"
_VolumeUnit = "ul"


def name_ligand(l: Molecule):  # kinda silly...
    for ll in _ligand_inventory:
        if ll == l:
            l.iupac_name = ll.iupac_name
            break
    return l


def load_robotinput_csv(
        ligand: Molecule,
        ligand_stock_solution_concentration: float,
        csv: FilePath,
) -> list[ReactionNcOneLigand]:
    reaction_df = pd.read_csv(csv)
    condition_df = reaction_df[["Reaction Parameters", "Parameter Values"]]
    condition_df = condition_df.dropna(axis=0, how="all")
    reaction_conditions = []
    for k, v in zip(condition_df.iloc[:, 0], condition_df.iloc[:, 1]):
        reaction_conditions.append(ReactionCondition(k, v))
    vial_col, solvent_col, ligand_col, nc_col = ["Vial Site", "Reagent1 (ul)", "Reagent2 (ul)", "Reagent6 (ul)"]
    try:
        reaction_df = reaction_df[[vial_col, solvent_col, ligand_col, nc_col]]
    except KeyError:
        # somehow `LS001_L4_robotinput` sheet mssing "Vial Site"
        reaction_df = reaction_df.rename(columns={" ": "Vial Site"})
        reaction_df = reaction_df[[vial_col, solvent_col, ligand_col, nc_col]]
    reactions = []
    for record in reaction_df.to_dict(orient="records"):
        vial = padding_vial_label(record[vial_col])
        solvent_volume = record[solvent_col]
        ligand_solution_volume = record[ligand_col]
        nc_solution_volume = record[nc_col]

        # skip invalid entries
        if solvent_volume < 1e-7 and nc_solution_volume < 1e-7 and ligand_solution_volume < 1e-7:
            continue

        identifier = get_basename(csv) + "--" + vial
        solvent = ReactantSolvent(_SolventIdentity, solvent_volume)
        ligand_solution = ReactantSolution(ligand, ligand_solution_volume, concentration=ligand_stock_solution_concentration,
                                  solvent_identity=_SolventIdentity, volume_unit=_VolumeUnit,
                                  concentration_unit=_LigandStockSolutionConcentrationUnit)
        nc = ReactantSolution("CsPbBr3", volume=nc_solution_volume, solvent_identity=_SolventIdentity,
                              volume_unit=_VolumeUnit,
                              concentration=None, properties={}, concentration_unit=None)
        reaction = ReactionNcOneLigand(identifier, nc, ligand_solution, reaction_conditions, solvent, properties={"vial": vial})
        reactions.append(reaction)
    return reactions


def load_fom_csv(
        ligands_used: list[Molecule],
        csv:FilePath,
):
    fom_dataframe = pd.read_csv(csv)
    layout_col = "Layout"
    fom_cols = [c for c in fom_dataframe.columns if c.endswith("PL_area")]
    dfom_cols = [c for c in fom_dataframe.columns if c.endswith("PL_area_s")]
    assert len(fom_cols) == len(dfom_cols) == len(ligands_used)
    data = dict()
    for i in range(len(ligands_used)):
        ligand_data = dict()
        ligand = ligands_used[i]
        fom_col = fom_cols[i]
        dfom_col = dfom_cols[i]
        this_df = fom_dataframe[[layout_col, fom_col, dfom_col]].dropna(axis=0, how="any")
        records = this_df.to_dict(orient="records")
        for r in records:
            ligand_data[padding_vial_label(r[layout_col])] = {"fom": r[fom_col], "dfom": r[dfom_col]}
        data[ligand] = ligand_data
    return data  # data[ligand][vial number] -> "fom":fom, "dfom":dfom


def plot_concentration_fom(real: [ReactionNcOneLigand], ref: [ReactionNcOneLigand], saveas: str):
    # TODO add error bars for dfom
    x_real, y_real, x_unit = reactions_to_xy(real)
    x_ref, y_ref, x_unit = reactions_to_xy(ref)
    fig, ax = plt.subplots()
    ax.scatter(x_real, y_real, label="sample")
    for iref, refval in enumerate(y_ref):
        if iref == 0:
            label = "reference"
        else:
            label = None
        ax.axhline(y=refval, xmin=0, xmax=1, c="gray", ls=":", label=label)
    ax.set_xlabel("Ligand Amount ({})".format(x_unit))
    ax.set_ylabel("Figure of Merit (a.u.)")
    ax.set_title(real[0].ligand.identity.iupac_name)
    ax.legend()
    fig.savefig("{}.png".format(saveas), dpi=300)


if __name__ == '__main__':
    LIGANDS_SPECIFICATION = """Name	CAS	Molecular Formula	InChI
    2,2′-Bipyridyl	366-18-7	C10H8N2	InChI=1S/C10H8N2/c1-3-7-11-9(5-1)10-6-2-4-8-12-10/h1-8H
    Lauric acid (dodecanoic acid)	143-07-7	C12H24O2	InChI=1S/C12H24O2/c1-2-3-4-5-6-7-8-9-10-11-12(13)14/h2-11H2,1H3,(H,13,14)
    Tributylamine	102-82-9	C12H27N	InChI=1S/C12H27N/c1-4-7-10-13(11-8-5-2)12-9-6-3/h4-12H2,1-3H3
    Octylphosphonic acid	4724-48-5	C8H19O3P	InChI=1S/C8H19O3P/c1-2-3-4-5-6-7-8-12(9,10)11/h2-8H2,1H3,(H2,9,10,11)
    Octadecanethiol	2885-00-9	C18H38S	InChI=1S/C18H38S/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19/h19H,2-18H2,1H3"""

    ligands_used = pd.read_table(StringIO(LIGANDS_SPECIFICATION))
    ligands_used = inventory_df_to_mols(ligands_used)
    ligands_used = [name_ligand(l) for l in ligands_used]

    reactions_large_1 = load_robotinput_csv(ligands_used[0], 0.05, "../data_csv/2022_0304_LS001_MK003-LS001_L1_robotinput.csv")
    reactions_large_2 = load_robotinput_csv(ligands_used[1], 0.05, "../data_csv/2022_0304_LS001_MK003-LS001_L2_robotinput.csv")
    reactions_large_3 = load_robotinput_csv(ligands_used[2], 0.05, "../data_csv/2022_0304_LS001_MK003-LS001_L3_robotinput.csv")
    reactions_large_4 = load_robotinput_csv(ligands_used[3], 0.05, "../data_csv/2022_0304_LS001_MK003-LS001_L4_robotinput.csv")
    reactions_large_5 = load_robotinput_csv(ligands_used[4], 0.05, "../data_csv/2022_0304_LS001_MK003-LS001_L5_robotinput.csv")

    reactions_small_1 = load_robotinput_csv(ligands_used[0], 100*1e-6, "../data_csv/2022_0322_LCL1_0021_robotinput-NIMBUS_reaction.csv")
    reactions_small_2 = load_robotinput_csv(ligands_used[1], 100*1e-6, "../data_csv/2022_0322_LCL2_0001_robotinput-NIMBUS_reaction.csv")
    reactions_small_3 = load_robotinput_csv(ligands_used[2], 100*1e-6, "../data_csv/2022_0323_LCL3_0014_robotinput-NIMBUS_reaction.csv")
    reactions_small_4 = load_robotinput_csv(ligands_used[3], 100*1e-6, "../data_csv/2022_0323_LCL4_0009_robotinput-NIMBUS_reaction.csv")
    reactions_small_5 = load_robotinput_csv(ligands_used[4], 100*1e-6, "../data_csv/2022_0323_LCL5_0008_robotinput-NIMBUS_reaction.csv")

    fom_large = load_fom_csv(ligands_used, "../data_csv/2022_0317_LS_PeakArea_mk-LS001_single_RPA_MK003.csv")
    fom_small = load_fom_csv(ligands_used, "../data_csv/2022_0323_LS_PeakArea_low_concentration_single_mk-LS003_LC_single_SB009.csv")

    for reaction_set in [reactions_large_1, reactions_large_2, reactions_large_3, reactions_large_4, reactions_large_5]:
        real, ref, blank = list(categorize_reactions(reaction_set).values())[0]
        for r in reaction_set:
            l = r.ligand.identity
            vial = r.properties["vial"]
            fom_ligand = fom_large[l]
            try:
                fom_ligand_vial = fom_ligand[vial]
            except KeyError:
                fom_ligand_vial = {"fom": 0, "dfom": 0}
            r.properties.update(fom_ligand_vial)

    for reaction_set in [reactions_small_1, reactions_small_2, reactions_small_3, reactions_small_4, reactions_small_5]:
        real, ref, blank = list(categorize_reactions(reaction_set).values())[0]
        for r in reaction_set:
            l = r.ligand.identity
            vial = r.properties["vial"]
            fom_ligand = fom_small[l]
            try:
                fom_ligand_vial = fom_ligand[vial]
            except KeyError:
                fom_ligand_vial = {"fom": 0, "dfom": 0}
            r.properties.update(fom_ligand_vial)

    def reactions_to_df(reactions_large, reactions_small):
        records = []
        real, ref, blank = list(categorize_reactions(reactions_large).values())[0]
        ref_fom = np.mean([r.properties["fom"] for r in ref])
        for r in real:
            amount = r.ligand.volume * r.ligand.concentration
            fom = r.properties["fom"]/ref_fom
            records.append({"ligand": r.ligand.identity.iupac_name, "amount": amount, "fom": fom, "window": "large"})

        real, ref, blank = list(categorize_reactions(reactions_small).values())[0]
        ref_fom = np.mean([r.properties["fom"] for r in ref])
        for r in real:
            amount = r.ligand.volume * r.ligand.concentration
            fom = r.properties["fom"]/ref_fom
            records.append({"ligand": r.ligand.identity.iupac_name, "amount": amount, "fom": fom, "window": "small"})
        return pd.DataFrame.from_records(records)

    l1_df = reactions_to_df(reactions_large_1, reactions_small_1)
    l2_df = reactions_to_df(reactions_large_2, reactions_small_2)
    l3_df = reactions_to_df(reactions_large_3, reactions_small_3)
    l4_df = reactions_to_df(reactions_large_4, reactions_small_4)
    l5_df = reactions_to_df(reactions_large_5, reactions_small_5)
    df = pd.concat([l1_df, l2_df, l3_df, l4_df, l5_df], axis=0)
    df.to_csv("output/df_windows.csv", index=False)

