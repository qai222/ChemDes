import numpy as np
import pandas as pd
from loguru import logger
from monty.json import MSONable
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

from lsal.schema import Molecule
from lsal.utils import scale_df

"""
Given a specific ranking parameter R and its values for a set of ligands L
select a diverse subset of L of top k percentile R

batch diversity following 
https://arxiv.org/pdf/1901.05954.pdf
"""


class DiversitySuggestor(MSONable):
    def __init__(
            self,
            suggestor_name: str,
            ranking_parameter: str,
            pool: dict[str, Molecule],
            ranking_dataframe: pd.DataFrame,
            percentile=2,
            percentile_from='top',
            batch_size=50,
            diversity_space='feature',
    ):
        """
        suggestor makes suggestions from a `LigandPool` by a `RankingParameter`:
        1. pre-filter `LigandPool` by keeping only the top or bottom x percentile, this gives the `FilteredPool`
        2. map `FilteredPool` to a space and run clustering to give n clusters of ligands
            by setting diversity_space=`feature`, the space is the feature space used in AL, kmeans will be used
            by setting diversity_space=`chemistry`, the space is MorganFP of size==1024, 
                Butina clustering (over Tanimoto distance) will be used
        3. within each cluster, the ligands are sorted by its distance to the centroid of the cluster
        """
        self.ranking_parameter = ranking_parameter
        self.suggestor_name = suggestor_name
        self.diversity_space = diversity_space
        self.percentile_from = percentile_from
        self.batch_size = batch_size
        self.percentile = percentile
        self.ranking_dataframe = ranking_dataframe
        self.pool = pool

        assert len(self.pool) == len(set(self.pool))
        assert all(m.is_featurized for m in self.pool.values())

        assert self.ranking_dataframe['ligand_label'].tolist() == [m.label for m in self.pool.values()]
        assert self.ranking_dataframe['ligand_identifier'].tolist() == [m.identifier for m in self.pool.values()]
        self.rank_methods = [c for c in ranking_dataframe.columns if c.startswith('rank_')]

    @property
    def details(self):
        original_rp_range = "[{:.4f}, {:.4f}]".format(
            min(self.ranking_dataframe[self.ranking_parameter]),
            max(self.ranking_dataframe[self.ranking_parameter])
        )
        explain = f"""
        {self.__class__.__name__}: {self.suggestor_name}
        >> Parameters
        ranking parameter is: {self.ranking_parameter}
        original ligand pool has size: {len(self.pool)}
        original ligand pool has ranking parameter range: {original_rp_range}
        prefilter ligand pool from: {self.percentile_from} {self.percentile}%
        on which space will clustering be performed?: {self.diversity_space}
        how many intended clusters?: {self.batch_size}
        """
        return explain.strip()

    def suggest(self):
        logger.info(f"SUGGEST based on self.ranking_parameter: {self.ranking_parameter}")
        logger.info(f"the pool size is: {len(self.pool)}")
        assert self.ranking_parameter in self.rank_methods
        logger.info(
            "ranking parameter range: [{:.4f}, {:.4f}]".format(
                min(self.ranking_dataframe[self.ranking_parameter]),
                max(self.ranking_dataframe[self.ranking_parameter])
            )
        )

        # prefilter based on percentile
        if self.percentile_from == 'top':
            cutoff_value = np.percentile(self.ranking_dataframe[self.ranking_parameter], 100 - self.percentile)
            rkdf = self.ranking_dataframe.loc[self.ranking_dataframe[self.ranking_parameter] >= cutoff_value]
            logger.info(f"use TOP percentile: {self.percentile}%")
        else:
            cutoff_value = np.percentile(self.ranking_dataframe[self.ranking_parameter], self.percentile)
            rkdf = self.ranking_dataframe.loc[self.ranking_dataframe[self.ranking_parameter] <= cutoff_value]
            logger.info(f"use BOTTOM percentile: {self.percentile}%")
        rkdf = rkdf[[c for c in rkdf.columns if c not in self.rank_methods] + [self.ranking_parameter, ]]
        pool = {k: self.pool[k] for k in rkdf['ligand_identifier']}
        assert len(pool) == len(rkdf)
        logger.info(f"cutoff value: {cutoff_value}")
        logger.info(f"size after pre-filtering: {len(rkdf)}")
        logger.info(
            "range after pre-filtering: [{:.4f}, {:.4f}]".format(
                min(rkdf[self.ranking_parameter]),
                max(rkdf[self.ranking_parameter])
            )
        )
        # batch_size should be <= # of ligands after pre-filtering
        assert self.batch_size <= len(pool)

        # on which space the diversity is defined?
        pool_list = [pool[k] for k in rkdf['ligand_identifier']]
        if self.diversity_space == 'feature':
            _, df = Molecule.l1_input(pool_list, amounts=None)
            feature_space_data = scale_df(df.select_dtypes('number'))
            # kmeans clustering
            kmeans = KMeans(n_clusters=self.batch_size, random_state=42)
            kmeans.fit(feature_space_data)
            assert kmeans.n_iter_ < kmeans.max_iter, f"kmeans may not be fully converged: {kmeans.n_iter_}/{kmeans.max_iter}"
            icluster_to_indices = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
            # ligands in each cluster is sorted by the distance between a ligand and the cluster center
            cluster_dfs = []
            for icluster, center_coord in enumerate(kmeans.cluster_centers_):
                cluster_indices = icluster_to_indices[icluster]
                sorted_cluster_indices = sorted(cluster_indices, key=lambda x: np.linalg.norm(
                    feature_space_data.values[x] - center_coord))
                cluster_df = rkdf.iloc[sorted_cluster_indices]
                cluster_dfs.append(cluster_df)

        elif self.diversity_space == 'chemistry':
            from rdkit import DataStructs
            from rdkit.ML.Cluster import Butina
            mols = [Chem.MolFromInchi(lig.identifier) for lig in pool_list]
            fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
            dists = []
            nfps = len(fps)
            distance_matrix_indices_translation = dict()
            for ii in range(nfps):
                distance_matrix_indices_translation[(ii, ii)] = 0

            for i in range(1, nfps):
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
                for j in range(i):
                    distance_matrix_indices_translation[(i, j)] = sims[j]
                    distance_matrix_indices_translation[(j, i)] = sims[j]
                dists.extend([1 - x for x in sims])
            # Butina clustering in rdkit is controlled by `distThresh`, run iterations to tune it so the final
            # number of clusters is similar to what we defined
            dist_thresh = 0.99
            dist_thresh_step = 0.001
            cs = Butina.ClusterData(dists, nfps, distThresh=dist_thresh, isDistData=True)
            logger.debug(f"init dist_thres == {dist_thresh}, # cluster == {len(cs)}")
            while len(cs) < self.batch_size:
                dist_thresh -= dist_thresh_step
                logger.debug(f"trying dist_thres == {dist_thresh}")
                cs = Butina.ClusterData(dists, nfps, distThresh=dist_thresh, isDistData=True)
                logger.debug(f"# of cluster by Butina: {len(cs)}")
            # The first element for each cluster is its centroid.
            cluster_dfs = []
            for ic, cindices in enumerate(cs):
                center_id = cindices[0]
                sorted_cindices = sorted(cindices, key=lambda x: distance_matrix_indices_translation[(center_id, x)])
                assert center_id == sorted_cindices[0]
                cluster_df = rkdf.iloc[sorted_cindices]
                cluster_dfs.append(cluster_df)

        else:
            raise ValueError(f"unknown diversity_space=={self.diversity_space}")
        return cluster_dfs
