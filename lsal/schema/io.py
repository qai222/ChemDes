import abc
import logging

import pandas as pd

from lsal.schema.material import Molecule
from lsal.schema.reaction import LigandExchangeReaction
from lsal.utils import FilePath, file_exists, get_extension


class FileLoader(abc.ABC):
    def __init__(
            self, name: str, allowed_format: list[str], desired_outputs: list[str],
            loaded=None,
    ):
        if desired_outputs is None:
            desired_outputs = []
        self.desired_outputs = desired_outputs
        self.loaded = loaded
        self.allowed_format = allowed_format
        self.name = name

    @abc.abstractmethod
    def load_file(self, *args, **kwargs):
        pass

    def pre_check(self, fn: FilePath):
        assert file_exists(fn), "cannot access or file does not exist: {}".format(fn)
        ext = get_extension(fn)
        assert ext in self.allowed_format, "extension not allowed: {} -- {}".format(ext, self.allowed_format)

    def post_check(self):
        pass

    def load(self, fn: FilePath, *args, **kwargs):
        logging.info("FILE LOADER: \n\t{}".format(self.__class__.__name__))
        logging.info("FILE LOADER DETAILS: \n\t{}".format(self.__class__.__doc__.strip()))
        logging.info("LOADING FILE: \n\t{}".format(fn))
        self.pre_check(fn)

        loaded = self.load_file(fn, *args, **kwargs)
        self.loaded = loaded
        logging.info("LOADED: \n\t{}".format("\n\t".join([d.__repr__() for d in self.loaded])))

        self.post_check()
        logging.info("LOADING FINISHED")
        return self.loaded


def get_ml_unknown_y_single_ligand(
        ligand_to_amounts: dict[Molecule, list[float]],
        ligand_to_des_record: dict[Molecule, dict],
):
    records = []
    df_ligands = []
    final_cols = set()
    for ligand, amounts in ligand_to_amounts.items():
        des_record = ligand_to_des_record[ligand]
        for amount in amounts:
            record = {"ligand_amount": amount}
            record.update(des_record)
            if len(final_cols) == 0:
                final_cols.update(set(record.keys()))
            records.append(record)
            df_ligands.append(ligand)
    df_x = pd.DataFrame.from_records(records, columns=sorted(final_cols))
    logging.info("ML INPUT:\n df_X: {}\t df_y: {}".format(df_x.shape, None))
    return df_ligands, df_x, None


def get_ml_known_y_single_ligand(
        ligand_to_reactions: dict[Molecule, list[LigandExchangeReaction]],
        ligand_to_des_record: dict[Molecule, dict],
        fom_def: str,
        fill_nan: bool = False,
):
    records = []
    df_ligands = []
    final_cols = set()
    for ligand, reactions in ligand_to_reactions.items():
        des_record = ligand_to_des_record[ligand]
        records_of_this_ligand = []
        for reaction in reactions:
            record = {
                "ligand_amount": reaction.ligand_solutions[0].amount,
                "FigureOfMerit": reaction.properties[fom_def],
            }
            record.update(des_record)
            if len(final_cols) == 0:
                final_cols.update(set(record.keys()))
            records_of_this_ligand.append(record)
            df_ligands.append(ligand)
        records += records_of_this_ligand
    df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
    df_x = df[[c for c in df.columns if c != "FigureOfMerit"]]
    df_y = df["FigureOfMerit"]
    if fill_nan:
        df_y.fillna(0, inplace=True)
    logging.info("ML INPUT:\n df_X: {}\t df_y: {}".format(df_x.shape, df_y.shape))
    return df_ligands, df_x, df_y
