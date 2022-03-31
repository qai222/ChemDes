import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsal.one_ligand import ReactionNcOneLigand, Molecule, ReactionCondition, ReactantSolvent, ReactantSolution, \
    reactions_to_xy, categorize_reactions
from lsal.schema import inventory_df_to_mols
from lsal.schema import load_inventory
from lsal.utils import strip_extension, json_dump

this_dir = os.path.abspath(os.path.dirname(__file__))

_ligand_inventory = load_inventory("../data/2022_0217_ligand_InChI_mk.xlsx", to_mols=True)
_SolventIdentity = "m-xylene"
_LigandStockSolutionConcentration = 0.05
_LigandStockSolutionConcentrationUnit = "M"
_VolumeUnit = "ul"


def padding_vial_label(v: str) -> str:
    """ pad zeros for vial label, e.g. A1 -> A01 """
    assert len(v) <= 3
    v_list = re.findall(r"[^\W\d_]+|\d+", v)
    try:
        assert len(v_list) == 2
        alpha, numeric = v_list
        assert len(numeric) <= 2
        final_string = alpha + "{0:02d}".format(int(numeric))
        assert len(final_string) == len(alpha) + 2
        return final_string
    except (AssertionError, ValueError) as e:
        raise ValueError("invalid vial: {}".format(v))


def name_ligand(l: Molecule):
    for ll in _ligand_inventory:
        if ll == l:
            l.iupac_name = ll.iupac_name
            break
    return l


def load_reaction_excel(ligand_specification_string: str, excel_path: Union[Path, str]) -> tuple[
    list[Molecule], list[ReactionNcOneLigand]]:
    # load excel file
    ef = pd.ExcelFile(os.path.join(this_dir, excel_path))

    # what are the ligands?
    ligands_used = pd.read_table(StringIO(ligand_specification_string))
    ligands_used = inventory_df_to_mols(ligands_used)
    ligands_used = [name_ligand(l) for l in ligands_used]

    # parse condition sheets
    reactions = []
    condition_sheets = ef.sheet_names[2:]
    assert all("robotinput" in sheet for sheet in condition_sheets)
    assert len(ligands_used) == len(condition_sheets)
    for ligand, sheet in zip(ligands_used, condition_sheets):
        condition_dataframe = ef.parse(sheet)
        reactions_of_this_ligand = load_reaction_dataframe(condition_dataframe, sheet, ligand)
        reactions += reactions_of_this_ligand
    return ligands_used, reactions


def load_reaction_dataframe(reaction_df: pd.DataFrame, sheet_name: str, ligand_mol: Molecule):
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

        identifier = sheet_name + "--" + vial
        solvent = ReactantSolvent(_SolventIdentity, solvent_volume)
        ligand = ReactantSolution(ligand_mol, ligand_solution_volume, concentration=_LigandStockSolutionConcentration,
                                  solvent_identity=_SolventIdentity, volume_unit=_VolumeUnit,
                                  concentration_unit=_LigandStockSolutionConcentrationUnit)
        nc = ReactantSolution("CsPbBr3", volume=nc_solution_volume, solvent_identity=_SolventIdentity,
                              volume_unit=_VolumeUnit,
                              concentration=None, properties={"batch": "MK003"}, concentration_unit=None)
        reaction = ReactionNcOneLigand(identifier, nc, ligand, reaction_conditions, solvent, properties={"vial": vial})
        reactions.append(reaction)
    return reactions


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


def default_warning(ligands: [Molecule]):
    logging.warning("=" * 3 + "CAREFULLY CHECK THE FOLLOWING DEFAULT SETTINGS !!!" + "=" * 3)
    logging.warning("The solvent is: {}".format(_SolventIdentity))
    logging.warning("The ligand stock solutions have concentration: {} {}".format(_LigandStockSolutionConcentration,
                                                                                  _LigandStockSolutionConcentrationUnit))
    logging.warning("The ligand solution was measured in: {}".format(_VolumeUnit))
    logging.warning("The ligands are, from L1 to L{}:\n{}".format(len(ligands), "\n".join(
        ["LIGAND: {}".format(m.iupac_name) for m in ligands])))
    logging.warning("=" * 3 + "DEFAULT SETTINGS END" + "=" * 3)


def categorized_reactions_warning(data: dict):
    for l in data:
        logging.warning(">> ligand: {}".format(l))
        real, ref, blank = data[l]
        logging.warning("# of real reactions: {}".format(len(real)))
        logging.warning("# of ref reactions: {}".format(len(ref)))
        logging.warning("# of blank reactions: {}".format(len(blank)))


def load_fom_sheet(excel_file: Union[Path, str], sheet_name: str, ligands_used: [Molecule]):
    ef = pd.ExcelFile(os.path.join(this_dir, excel_file))
    fom_dataframe = ef.parse(sheet_name)

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


if __name__ == '__main__':
    logging.basicConfig(filename='{}.log'.format(strip_extension(os.path.basename(__file__))), filemode="w")

    LIGANDS_SPECIFICATION = """Name	CAS	Molecular Formula	InChI
    2,2′-Bipyridyl	366-18-7	C10H8N2	InChI=1S/C10H8N2/c1-3-7-11-9(5-1)10-6-2-4-8-12-10/h1-8H
    Lauric acid (dodecanoic acid)	143-07-7	C12H24O2	InChI=1S/C12H24O2/c1-2-3-4-5-6-7-8-9-10-11-12(13)14/h2-11H2,1H3,(H,13,14)
    Tributylamine	102-82-9	C12H27N	InChI=1S/C12H27N/c1-4-7-10-13(11-8-5-2)12-9-6-3/h4-12H2,1-3H3
    Octylphosphonic acid	4724-48-5	C8H19O3P	InChI=1S/C8H19O3P/c1-2-3-4-5-6-7-8-12(9,10)11/h2-8H2,1H3,(H2,9,10,11)
    Octadecanethiol	2885-00-9	C18H38S	InChI=1S/C18H38S/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19/h19H,2-18H2,1H3"""

    ligands_used, reactions = load_reaction_excel(LIGANDS_SPECIFICATION,
                                                  excel_path="../data/2022_0304_LS001_MK003.xlsx")
    default_warning(ligands_used)

    fom_data = load_fom_sheet("../data/2022_0317_LS_PeakArea_mk.xlsx", sheet_name="LS001_single_RPA_MK003",
                              ligands_used=ligands_used)

    invalid_reactions = []
    for ir, r in enumerate(reactions):
        try:
            entry = fom_data[r.ligand.identity][r.properties["vial"]]
            r.properties["fom"] = entry["fom"]
            r.properties["dfom"] = entry["dfom"]
        except KeyError:
            logging.critical(
                "this reaction is missing FOM: {}".format(r.ligand.identity.iupac_name + "--" + r.properties["vial"]))
            invalid_reactions.append(ir)
    for i in invalid_reactions:
        reactions.pop(i)

    ligand_to_categorized_reactions = categorize_reactions(reactions)
    categorized_reactions_warning(ligand_to_categorized_reactions)

    # plot c vs fom
    for l in ligand_to_categorized_reactions:
        real, ref, blank = ligand_to_categorized_reactions[l]
        ref_fom = np.mean([r.properties["fom"] for r in ref])
        for r in real + ref + blank:
            try:
                r.properties["fom"] /= ref_fom
            except KeyError:
                continue
        plot_concentration_fom(real, ref, saveas="c_vs_fom/" + l.iupac_name)

    json_dump({k.__repr__(): v for k, v in ligand_to_categorized_reactions.items()},
              "output/2022_0304_LS001_MK003_reaction_data.json")
