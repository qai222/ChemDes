from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from loguru import logger
from monty.json import MSONable

from lsal.campaign.loader import load_molecules
from lsal.schema.material import Molecule, featurize_molecules
from lsal.utils import FilePath


class LigandPool(MSONable):
    def __init__(self, molecules: list[Molecule], ):
        # the pool should not have duplicates
        assert len(molecules) == len(set([m.label for m in molecules])) == len(set([m.identifier for m in molecules]))
        self.molecules = tuple(sorted(molecules))

    def __repr__(self):
        return f"{self.__class__.__name__}: size=={len(self)}"

    def __iter__(self):
        return self.molecules.__iter__()

    @property
    def identifiers(self):
        return tuple([m.identifier for m in self.molecules])

    def __getitem__(self, item):
        return self.molecules[item]

    def __len__(self):
        return len(self.molecules)

    def __hash__(self):
        return hash(self.identifiers)

    def __eq__(self, other):
        assert isinstance(other, LigandPool)
        return self.identifiers == other.identifiers

    def remove_one(self, m: Molecule) -> LigandPool:
        return LigandPool([mm for mm in self.molecules if mm.identifier != m.identifier])

    def to_feature_dataframe(self, ligand_amounts=None) -> pd.DataFrame:
        """
        use this to generate model input dataframe

        :param ligand_amounts:
            if None, dataframe.shape[0] == len(self)
            if a list/np.ndarray of float, dataframe.shape[0] == len(self) * len(ligand_amounts)
            if a dict[ligand, float], dataframe.shape[0] == $\Sum_i len(ligand_amounts[ligand_i])$
        :return: model input dataframe
        """
        records = []
        for m in self.molecules:
            if ligand_amounts is None:
                record = m.properties['features']
                records.append(record)
            elif isinstance(ligand_amounts, list) or isinstance(ligand_amounts, np.ndarray):
                for lig_amount in ligand_amounts:
                    record = deepcopy(m.properties['features'])
                    record['ligand_amount'] = lig_amount
                    records.append(record)
            elif isinstance(ligand_amounts, dict) or isinstance(ligand_amounts, OrderedDict):
                for lig_amount in ligand_amounts[m]:
                    record = deepcopy(m.properties['features'])
                    record['ligand_amount'] = lig_amount
                    records.append(record)
        return pd.DataFrame.from_records(records)

    @classmethod
    def from_csvs(
            cls, inv_csv: FilePath, des_csv: FilePath,
            col_to_mol_kw: dict[str, str], mol_type: str,
    ):
        molecules = load_molecules(inv_csv, col_to_mol_kw, mol_type)
        logger.info(f"Loaded molecules from: {inv_csv}")
        feature_df = pd.read_csv(des_csv)
        logger.info(f"Loaded features from: {des_csv}")
        featurize_molecules(molecules, feature_df)
        return cls(molecules)
