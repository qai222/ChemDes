import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chemdes.one_ligand import ReactionNcOneLigand, Molecule, ReactionCondition, ReactantSolvent, ReactantSolution, \
    reactions_to_xy, categorize_reactions
from chemdes.schema import inventory_df_to_mols
from chemdes.schema import load_inventory
from chemdes.utils import json_dump, strip_extension

this_dir = os.path.abspath(os.path.dirname(__file__))

_SolventIdentity = "m-xylene"
_LigandStockSolutionConcentration = 0.05
_LigandStockSolutionConcentrationUnit = "M"
_VolumeUnit = "ul"


class FomStrategy:

    # @staticmethod
    # def FOM_strategy_BINARY_BetterThanSmallestRef(ref_reactions: [ReactionNcOneLigand],
    #                                               real_reactions: [ReactionNcOneLigand]):
    #     """ is there at least one fom higher than the smallest ref fom? """
    #     min_ref_fom = min([r.properties["fom"] for r in ref_reactions])
    #     max_real_fom = max([r.properties["fom"] for r in real_reactions])
    #     return int(max_real_fom > min_ref_fom)

    @staticmethod
    def FOM_strategy_CONTINUOUS_MaxFom(ref_reactions, real_reactions: [ReactionNcOneLigand]):
        """ max fom of real reactions """
        return max([r.properties["fom"] for r in real_reactions])

    @staticmethod
    def FOM_strategy_CONTINUOUS_AvgTop5(ref_reactions, real_reactions: [ReactionNcOneLigand]):
        """ avg of top 5 fom of real reactions """
        return np.mean(sorted([r.properties["fom"] for r in real_reactions], reverse=True)[:5])


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


def load_reactions(ligand_mols, reaction_path: Union[Path, str]):
    # load excel file
    ef = pd.ExcelFile(os.path.join(this_dir, reaction_path))

    # parse the figure of merit sheet
    fom_sheet = ef.sheet_names[1]
    fom_df = ef.parse(fom_sheet)
    vial_col = "Layout"
    assert len(fom_df.columns == len(ligand_mols) + 1)
    assert fom_df.columns[0] == vial_col
    fom_dict = dict()
    for iligand, ligand_fom_col in enumerate(fom_df.columns[1:]):
        ligand_mol = ligand_mols[iligand]
        fom_dict[ligand_mol] = dict()
        for vial, fom_val in zip(fom_df[vial_col], fom_df[ligand_fom_col]):
            fom_dict[ligand_mol][padding_vial_label(vial)] = fom_val

    # parse the conditions
    all_reactions = []
    ligand_sheets = ef.sheet_names[2:]
    assert len(ligand_sheets) == len(ligand_mols)
    for ligand_mol, sheet_name in zip(ligand_mols, ligand_sheets):
        reaction_dataframe = ef.parse(sheet_name)
        reactions = load_reaction_dataframe(reaction_dataframe, sheet_name, ligand_mol)
        # link each reaction to a fom
        for r in reactions:
            reaction_fom = fom_dict[ligand_mol][r.properties["vial"]]
            r.properties["fom"] = reaction_fom
        all_reactions += reactions

    # # check reactions
    # for r in all_reactions:
    #     r.check()
    return all_reactions


def plot_concentration_fom(real: [ReactionNcOneLigand], ref: [ReactionNcOneLigand], saveas: str):
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
    logging.basicConfig(filename='{}.log'.format(strip_extension(os.path.basename(__file__))), filemode="w")

    ligands_from_inventory = load_inventory("../data/2022_0217_ligand_InChI_mk.xlsx", to_mols=True)

    # read ligand information from the first sheet of 2022_0304_LS001_MK003
    Ligand_Selected = StringIO("""Name	CAS	Molecular Formula	InChI
    2,2′-Bipyridyl	366-18-7	C10H8N2	InChI=1S/C10H8N2/c1-3-7-11-9(5-1)10-6-2-4-8-12-10/h1-8H
    Lauric acid (dodecanoic acid)	143-07-7	C12H24O2	InChI=1S/C12H24O2/c1-2-3-4-5-6-7-8-9-10-11-12(13)14/h2-11H2,1H3,(H,13,14)
    Tributylamine	102-82-9	C12H27N	InChI=1S/C12H27N/c1-4-7-10-13(11-8-5-2)12-9-6-3/h4-12H2,1-3H3
    Octylphosphonic acid	4724-48-5	C8H19O3P	InChI=1S/C8H19O3P/c1-2-3-4-5-6-7-8-12(9,10)11/h2-8H2,1H3,(H2,9,10,11)
    Octadecanethiol	2885-00-9	C18H38S	InChI=1S/C18H38S/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19/h19H,2-18H2,1H3""")
    df_ligand = pd.read_table(Ligand_Selected)
    ligands = inventory_df_to_mols(df_ligand)

    # add iupac_names
    for l in ligands:
        for ll in ligands_from_inventory:
            if l == ll:
                l.iupac_name = ll.iupac_name

    # check defaults
    logging.warning("=" * 3 + "CAREFULLY CHECK THE FOLLOWING DEFAULT SETTINGS !!!" + "=" * 3)
    logging.warning("The solvent is: {}".format(_SolventIdentity))
    logging.warning("The ligand stock solutions have concentration: {} {}".format(_LigandStockSolutionConcentration,
                                                                                  _LigandStockSolutionConcentrationUnit))
    logging.warning("The ligand solution was measured in: {}".format(_VolumeUnit))
    logging.warning("The ligands are, from L1 to L{}:\n{}".format(len(ligands), "\n".join(
        ["LIGAND: {}".format(m.iupac_name) for m in ligands])))
    logging.warning("=" * 3 + "DEFAULT SETTINGS END" + "=" * 3)

    # read reactions
    logging.warning("...reading reactions...")
    reactions = load_reactions(ligands, "../data/2022_0304_LS001_MK003_with_update.xlsx")

    # subgroup reactions by ligand, reaction type (ref, blank, or real)
    reaction_data = []
    ligand_to_categorized_reactions = categorize_reactions(reactions)
    for ligand, categorized_reactions in categorize_reactions(reactions).items():
        # data holder for this ligand
        ligand_data = dict()
        ligand_data["ligand"] = ligand

        real_reactions, ref_reactions, blank_reactions = categorized_reactions
        ligand_data["real_reactions"] = real_reactions
        ligand_data["ref_reactions"] = ref_reactions
        ligand_data["blank_reactions"] = blank_reactions

        # check blank reactions
        for r in blank_reactions:
            try:
                assert r.properties["fom"] < 1e-7 or np.isnan(r.properties["fom"])
            except AssertionError:
                logging.critical("this BLANK reaction has nonzero FOM???")
                logging.critical(r.__repr__())

        # plot c vs fom for each ligand
        plot_concentration_fom(real_reactions, ref_reactions, "c_vs_fom/" + ligand.iupac_name)

        # fom value for a ligand
        fs = FomStrategy()
        for strategy in [getattr(fs, n) for n in dir(fs) if n.startswith("FOM_")]:
            fom_value = strategy(ref_reactions, real_reactions)
            ligand_data[strategy.__name__] = fom_value

        # printout info
        logging.warning("ligand name: {}".format(ligand.iupac_name))
        logging.warning("# of real reactions: {}".format(len(real_reactions)))
        logging.warning("# of blank reactions: {}".format(len(blank_reactions)))
        logging.warning("# of ref reactions: {}".format(len(ref_reactions)))

        reaction_data.append(ligand_data)

    # export json
    json_dump(reaction_data, "output/2022_0304_LS001_MK003_reaction_data.json")
