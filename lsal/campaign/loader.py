from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from monty.json import MSONable

from lsal.schema import FileLoader, Molecule, ReactionCondition, LigandExchangeReaction, \
    ReactantSolution, NanoCrystal, _EPS
from lsal.utils import FilePath, padding_vial_label, strip_extension, get_extension


class LoaderInventory(FileLoader):
    """
    load raw ligand file (a spreadsheet) to list[Molecule]
    """

    def __init__(self, name: str, mol_type: str, loaded: dict = None, ):
        mol_type = mol_type.upper()
        super().__init__(name, ["xlsx", "csv"], ["{}_inventory".format(mol_type), ], loaded)
        self.mol_type = mol_type

    def load_file(
            self, fn: FilePath,
            inventory_columns=("Name", "IUPAC Name", "InChI",),
    ) -> list[Molecule]:
        _, extension = os.path.splitext(fn)
        if extension == ".csv":
            df = pd.read_csv(fn)
        elif extension == ".xlsx":
            ef = pd.ExcelFile(fn)
            assert len(ef.sheet_names) == 1, "there should be only one sheet in the xlsx file"
            df = ef.parse(ef.sheet_names[0])
        else:
            raise AssertionError("inventory file should be either csv or xlsx")
        assert set(inventory_columns).issubset(set(df.columns)), "Inventory DF does not have required columns"
        df = df.dropna(axis=0, how="all", subset=inventory_columns)
        molecules = []
        for irow, row in enumerate(df.to_dict("records")):
            name = row["Name"]
            iupac_name = row["IUPAC Name"]
            inchi = row["InChI"]
            m = Molecule(
                identifier=inchi, iupac_name=iupac_name, name=name, int_label=irow, mol_type=self.mol_type,
                properties={"raw_file": os.path.basename(fn)}
            )
            molecules.append(m)
        return molecules


class LoaderLigandDescriptors(FileLoader):
    # TODO maybe a dedicated featureization class...
    """
    load ligand descriptor file (a csv) to dict[Ligand, DescriptorRecord]
    """

    def __init__(self, name: str, loaded: dict = None):
        super().__init__(name, ["csv"], ["ligand_to_descriptors"], loaded)

    def load_file(self, fn: FilePath, inventory: list[Molecule]) -> dict[Molecule, dict]:
        ligands = []
        des_df = pd.read_csv(fn)
        assert not des_df.isnull().values.any()
        descriptor_records = des_df.to_dict("records")
        for r in descriptor_records:
            inchi = r["InChI"]
            try:
                mol = Molecule.select_from_inventory(inchi, inventory, "inchi")
                mol: Molecule
                ligands.append(mol)
            except ValueError:
                logging.critical("molecule not found in the inventory, skipping: {}".format(inchi))
                continue
        data_df = des_df.select_dtypes('number')
        descriptor_records = data_df.to_dict("records")
        descriptor_records: list[dict[str, Any]]
        return dict(zip(ligands, descriptor_records))


class LoaderPeakInfo(FileLoader):
    """
    load peak into file to dict[vial_identifier, peak_data_dict]
    """

    def __init__(self, name: str, loaded: dict = None):
        super().__init__(name, ["csv"], ["peak_data"], loaded)

    def load_file(self, fn: FilePath, identifier_prefix: str, vial_column: str = "wellLabel") -> dict[str, dict]:
        peak_df = pd.read_csv(fn)
        peak_df.drop(peak_df.filter(regex="Unnamed"), axis=1, inplace=True)
        peak_df.dropna(axis=0, inplace=True, how="all")

        vial_col = [c for c in peak_df.columns if vial_column in c]
        assert len(vial_col) == 1, "`{}` can only appear once in the columns of: {}".format(vial_column, fn)
        vial_col = vial_col[0]

        data = dict()
        for record in peak_df.to_dict(orient="records"):
            vial = identifier_prefix + "@@" + padding_vial_label(record[vial_col])
            data[vial] = dict()
            data[vial]["peak_file"] = os.path.basename(fn)
            data[vial].update(record)
        return data


class LoaderRobotInputSlc(FileLoader):
    """ load robotinput to ligand reactions """

    _DefaultRobotInputColumns = (
        'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)',
        'Reagent5 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)',
        'Reagent10 (ul)', 'Labware ID:', 'Reaction Parameters', 'Parameter Values', 'Reagents',
        'Reagent Name', 'Reagent Identity', 'Reagent Concentration (uM)', 'Liquid Class',
    )
    _DefaultReagentColumns = ("Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)")
    _VialCol = _DefaultRobotInputColumns[0]
    _ConcentrationUnit = "uM"
    _VolumeUnit = "ul"
    _DefaultSolvents = ("m-xylene",)

    def __init__(self, name: str, loaded: dict = None):
        super().__init__(name, ["csv", "xls"], ["reactions"], loaded)

    def load_file(
            self, fn: FilePath,
            ligand_inventory: list[Molecule],
            solvent_inventory: list[Molecule],
            input_columns: list[str] = _DefaultRobotInputColumns,
            reagent_columns: list[str] = _DefaultReagentColumns,
            vial_column: str = _VialCol,
            volume_unit: str = _VolumeUnit,
            concentration_unit: str = _ConcentrationUnit,
            possible_solvent: list[str] = _DefaultSolvents,
    ) -> list[LigandExchangeReaction]:
        """
        load robotinput to ligand reactions

        :param fn: robotinput file
        :param solvent_inventory:
        :param ligand_inventory: molecules in the inventory,
                labels in the input file will be used to map reagents to ligands
        :param input_columns: useful columns in the robotinput file
        :param reagent_columns: columns regarding reagents used
        :param vial_column: columns used to identify each reaction (vial)
        :param volume_unit: e.g. "ul"
        :param concentration_unit: e.g. "uM"
        :param possible_solvent: list of low case solvent names
        :return: a list of reactions
        """
        robot_input_name = strip_extension(os.path.basename(fn))
        ext = get_extension(fn)
        if ext == "csv":
            robotinput_df = pd.read_csv(fn)
        elif ext == "xls":
            robotinput_df = pd.read_excel(fn, 0)
        else:
            raise NotImplementedError

        # remove excess cells
        robotinput_df.drop(robotinput_df.filter(regex="Unnamed"), axis=1, inplace=True)
        robotinput_df.dropna(axis=0, inplace=True, how="all")

        # sanity check
        assert set(reagent_columns).issubset(
            set(input_columns)), "`reagent_columns` is not a subset of `input_columns`!"
        assert set(robotinput_df.columns) == set(
            input_columns), "`input_columns` is not identical to what we read from: {}".format(fn)

        # load general reaction conditions
        condition_df = robotinput_df.loc[:, ["Reaction Parameters", "Parameter Values"]]
        condition_df = condition_df.dropna(axis=0, how="all")
        reaction_conditions = []
        for k, v in zip(condition_df.iloc[:, 0], condition_df.iloc[:, 1]):
            rc = ReactionCondition(k, v)
            reaction_conditions.append(rc)

        # load reagents
        reagent_df = robotinput_df.loc[:, reagent_columns]
        reagent_df.dropna(axis=0, inplace=True, how="all")
        reagent_index_to_reactant = self.reagent_df_parser(
            df=reagent_df, ligand_inventory=ligand_inventory,
            solvent_inventory=solvent_inventory,
            volume_unit=volume_unit,
            concentration_unit=concentration_unit,
            used_solvents=possible_solvent,
        )

        # load volumes
        volume_df = robotinput_df.loc[:, ["Vial Site", ] + list(reagent_index_to_reactant.keys())]
        reactions = []
        for record in volume_df.to_dict(orient="records"):
            # vial
            logging.info("loading reaction: {}".format(record))
            vial = padding_vial_label(record[vial_column])
            identifier = "{}@@{}".format(robot_input_name, vial)

            ligand_reactants = []
            solvent_reactant, nc_reactant = None, None
            for reagent_index, reactant in reagent_index_to_reactant.items():
                volume = record[reagent_index]
                if volume < _EPS:
                    continue
                actual_reactant = deepcopy(reactant)
                actual_reactant.volume = volume
                reactant_def = actual_reactant.properties["definition"]
                if reactant_def == "nc":
                    nc_reactant = actual_reactant
                elif reactant_def == "ligand_solution":
                    ligand_reactants.append(actual_reactant)
                elif reactant_def == "solvent":
                    solvent_reactant = actual_reactant
                else:
                    raise ValueError("wrong definition: {}".format(reactant_def))

            reaction = LigandExchangeReaction(
                identifier=identifier, conditions=reaction_conditions, solvent=solvent_reactant,
                nc_solution=nc_reactant, ligand_solutions=ligand_reactants, properties=None,
            )
            reactions.append(reaction)
        return reactions

    @staticmethod
    def reagent_df_parser(
            df: pd.DataFrame, ligand_inventory: list[Molecule], solvent_inventory: list[Molecule],
            volume_unit: str, concentration_unit: str, used_solvents: list[str],
    ) -> dict[str, ReactantSolution]:
        solvent_material = None
        reagent_index_to_reactant = dict()
        for record in df.to_dict(orient="records"):
            reagent_index = record["Reagents"]
            reagent_name = record["Reagent Name"]
            reagent_identity = record["Reagent Identity"]
            reagent_concentration = record["Reagent Concentration (uM)"]

            if reagent_name.startswith("CPB") and pd.isnull(reagent_concentration):
                logging.info("Found nanocrystal: {}".format(reagent_name))
                material = NanoCrystal(identifier=reagent_name)
                reactant = ReactantSolution(solute=material, volume=np.nan, concentration=None, solvent=None,
                                            properties={"definition": "nc"}, volume_unit=volume_unit,
                                            concentration_unit=concentration_unit, )
                reagent_index_to_reactant[reagent_index] = reactant
            elif reagent_name.lower() in used_solvents and pd.isnull(reagent_concentration):
                solvent_name = reagent_name.lower()
                solvent_material = Molecule.select_from_inventory(solvent_name, solvent_inventory, "name")
                reactant = ReactantSolution(
                    solute=solvent_material, volume=np.nan, concentration=0.0, solvent=solvent_material,
                    properties={"definition": "solvent"},
                    volume_unit=volume_unit, concentration_unit=concentration_unit,
                )
            else:
                reagent_int_label = int(reagent_identity)
                ligand_molecule = Molecule.select_from_inventory(reagent_int_label, ligand_inventory, "int_label")
                reactant = ReactantSolution(
                    solute=ligand_molecule, volume=np.nan, concentration=reagent_concentration, solvent=None,
                    properties={"definition": "ligand_solution"},
                    volume_unit=volume_unit, concentration_unit=concentration_unit
                )
            reagent_index_to_reactant[reagent_index] = reactant

        for reagent_index, reactant in reagent_index_to_reactant.items():
            reactant.solvent = solvent_material
        return {"{} (ul)".format(k): v for k, v in reagent_index_to_reactant.items()}
        # note the "ul" in keys here are hardcoded for robotinput files


class ReactionCollection(MSONable):
    # TODO the reactions in a collection should have something in common (e.g. solvent/mixing conditions)
    def __init__(self, reactions: list[LigandExchangeReaction], properties: dict = None):
        self.reactions = reactions
        if properties is None:
            properties = dict()
        self.properties = properties

    @property
    def ligand_amount_range(self):
        amounts = []
        amount_unit = []
        for r in self.real_reactions:
            for ls in r.ligand_solutions:
                amounts.append(ls.amount)
                amount_unit.append(ls.amount_unit)
        assert len(set(amount_unit)) == 1
        return min(amounts), max(amounts), amount_unit[0]

    @classmethod
    def subset_by_lcombs(cls, campaign_reactions: ReactionCollection, lc_subset):
        reactions = [r for r in campaign_reactions.real_reactions if r.ligands in lc_subset]
        return cls(reactions, properties=deepcopy(campaign_reactions.properties))

    @property
    def real_reactions(self):
        reactions = []
        for r in self.reactions:
            if r.is_reaction_real:
                reactions.append(r)
            else:
                continue
        return reactions

    @property
    def unique_lcombs(self):
        lcombs = set()
        for r in self.real_reactions:
            lcombs.add(r.unique_ligands)
        return sorted(lcombs)

    def __repr__(self):
        s = "{}\n".format(self.__class__.__name__)
        s += "\t# of reactions: {}\n".format(len(self.reactions))
        return s

    def get_lcomb_to_reactions(self, limit_to=None):
        reactions = self.real_reactions

        lcombs, grouped_reactions = LigandExchangeReaction.group_reactions(reactions, field="unique_ligands")
        lcombs_to_reactions = dict(zip(lcombs, grouped_reactions))
        if limit_to is None:
            return lcombs_to_reactions
        else:
            return {c: lcombs_to_reactions[c] for c in limit_to}

    @classmethod
    def from_files(
            cls,

            experiment_input_files: list[FilePath],
            experiment_output_files: list[FilePath],

            ligand_inventory: list[Molecule],
            solvent_inventory: list[Molecule],

            input_loader_kwargs: dict = None,
            output_loader_kwargs: dict = None,
            properties: dict = None,
    ):
        input_loader = LoaderRobotInputSlc("input_loader")
        output_loader = LoaderPeakInfo("output_loader")

        if input_loader_kwargs is None:
            input_loader_kwargs = {"ligand_inventory": ligand_inventory, "solvent_inventory": solvent_inventory}
        if output_loader_kwargs is None:
            output_loader_kwargs = {}

        all_reactions = []
        for ifile, ofile in zip(experiment_input_files, experiment_output_files):
            reactions = input_loader.load(ifile, **input_loader_kwargs)
            peak_data = output_loader.load(ofile, identifier_prefix=strip_extension(os.path.basename(ifile)),
                                           **output_loader_kwargs)
            cls.assign_reaction_results(reactions, peak_data)
            all_reactions += reactions

        # remove suspicious reactions
        n_real = 0
        n_blank = 0
        n_ref = 0
        for r in all_reactions:
            r: LigandExchangeReaction
            if r.is_reaction_blank_reference:
                n_blank += 1
            elif r.is_reaction_nc_reference:
                n_ref += 1
            elif r.is_reaction_real:
                n_real += 1
            else:
                raise Exception("reaction type cannot be determined: {}".format(r))
        logging.warning("REACTIONS LOADED: blank/ref/real: {}/{}/{}".format(n_blank, n_ref, n_real))
        slc = cls(reactions=all_reactions, properties=properties)
        slc.properties.update(
            dict(experiment_input_files=experiment_input_files, experiment_output_files=experiment_output_files,
                 ligand_inventory=ligand_inventory, solvent_inventory=solvent_inventory,
                 input_loader_kwargs=input_loader_kwargs, output_loader_kwargs=output_loader_kwargs, ))
        return slc

    @staticmethod
    def assign_reaction_results(reactions: list[LigandExchangeReaction], peak_data: dict[str, dict]):
        assert len(peak_data) == len(reactions)
        for r in reactions:
            data = peak_data[r.identifier]
            r.properties.update(data)
