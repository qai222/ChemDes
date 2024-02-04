import typing

import numpy as np
import pandas as pd
import umap
from loguru import logger

from lsal.alearn.one_ligand_worker import Worker, log_time
from lsal.schema import Molecule
from lsal.utils import get_basename, get_workplace_data_folder, get_folder, file_exists, similarity_matrix
from lsal.utils import json_load, FilePath, calculate_distance_matrix, SEED
"""
dimensionality reduction visualization for the molecular pool
only used in visualization
"""


_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)


class DimRed(Worker):
    def __init__(
            self,
            code_dir: FilePath,
            work_dir: FilePath,
            ligands_json_gz: FilePath,

            # calculations based on fp similarity
            fp_type: str = 'ECFP4',
            dimred_params: typing.Tuple[str] = ('nn=50;md=0.3', 'nn=50;md=0.7'),
    ):
        super().__init__(name=self.__class__.__name__, code_dir=code_dir, work_dir=work_dir)
        self.dimred_params = dimred_params
        self.ligands_json_gz = ligands_json_gz
        self.fp_type = fp_type

        self.ligands = json_load(self.ligands_json_gz)
        self.ligands: list[Molecule]
        assert len(self.ligands) == len(set(self.ligands))

        # tmp data to save time
        self.tmp_dmat_feat_npy = f'{self.work_dir}/dmat_feat.npy'
        self.tmp_dmat_chem_npy = f'{self.work_dir}/dmat_chem.npy'

        self.output = f'{self.code_dir}/df_dimred.csv'

    @log_time
    def load_df_dimred(self):
        columns = []
        values = []
        for space in ['FEAT', 'CHEM']:
            if space == 'FEAT':
                tmp_dmat_npy = self.tmp_dmat_feat_npy
            elif space == 'CHEM':
                tmp_dmat_npy = self.tmp_dmat_chem_npy
            else:
                raise ValueError

            if file_exists(tmp_dmat_npy):
                logger.warning(f"found previously saved distance matrix: {tmp_dmat_npy}")
                with open(tmp_dmat_npy, 'rb') as f:
                    dmat = np.load(f)
            else:
                logger.info(f"cannot find previously saved distance matrix at {tmp_dmat_npy}")
                if space == 'FEAT':
                    _, feature_space = Molecule.l1_input(self.ligands, amounts=None)
                    dmat = calculate_distance_matrix(feature_space, "manhattan")
                elif space == 'CHEM':
                    dmat = similarity_matrix([lig.smiles for lig in self.ligands], fp_type=self.fp_type)
                else:
                    raise ValueError
                with open(tmp_dmat_npy, 'wb') as f:
                    np.save(f, dmat)

            assert dmat.shape == (len(self.ligands), len(self.ligands))

            values_space = []
            for nnmd in self.dimred_params:
                nn, md = nnmd.split(";")
                nn = int(nn.replace("nn=", ""))
                md = float(md.replace("md=", ""))
                transformer = umap.UMAP(
                    n_neighbors=nn, min_dist=md, metric="precomputed", random_state=SEED)
                data_2d = transformer.fit_transform(dmat)
                values_space.append(data_2d)
                # to comply with mongo convention
                columns += [f"DIMRED_{space}_{nnmd}_x".replace(".", "[dot]"),
                            f"DIMRED_{space}_{nnmd}_y".replace(".", "[dot]"), ]
            values_space = np.concatenate(values_space, axis=1)
            assert values_space.shape == (len(self.ligands), 2 * len(self.dimred_params))
            values.append(values_space)
        values = np.concatenate(values, axis=1)
        assert values.shape == (len(self.ligands), 2 * 2 * len(self.dimred_params))
        dimred_df = pd.DataFrame(values, columns=columns)
        dimred_df['ligand_label'] = [li.label for li in self.ligands]
        dimred_df.set_index('ligand_label')
        dimred_df.to_csv(self.output)


if __name__ == '__main__':
    worker = DimRed(
        code_dir=_code_folder,
        work_dir=_work_folder,
        ligands_json_gz=f"{_code_folder}/../../MolecularInventory/ligands.json.gz",
        fp_type="ECFP4",
    )
    worker.run(
        ['load_df_dimred', ]
    )
