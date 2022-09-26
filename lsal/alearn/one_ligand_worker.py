import glob

import ChemScraper
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from lsal.alearn import SingleLigandLearner, SingleLigandPrediction, QueryRecord
from lsal.schema import Worker, Molecule, L1XReactionCollection
from lsal.tasks import MoleculeSampler
from lsal.utils import log_time, json_load, json_dump, pkl_dump, \
    pkl_load, chunks, createdir, calculate_distance_matrix, cut_end


def ks_sample_ligand_dataframe(
        df: pd.DataFrame, id_to_lig: dict[str, Molecule], id_field='ligand_identifier', sample_size: int = 40
):
    ligand_col = [id_to_lig[i] for i in df[id_field]]
    _, descriptor_df = Molecule.l1_input(ligand_col, amounts=None)
    ms = MoleculeSampler(ligand_col, dmat=calculate_distance_matrix(descriptor_dataframe=descriptor_df))
    indices = ms.sample_ks(k=sample_size, return_mol=False)
    return df.iloc[indices]


def get_shortlist(
        ranking_dataframe: pd.DataFrame, rank_method: str,
        cut_length: int, ks_sample_size: int,
        id_to_lig: dict[str, Molecule],
        use_head: bool, delta_cutoff: float = None,
        already_taught_label: list[str] = None,
) -> pd.DataFrame:
    ligand_fields = [m for m in ranking_dataframe.columns if m.startswith("ligand_")]
    df = ranking_dataframe.sort_values(by=rank_method, inplace=False, ascending=False)
    df = df[ligand_fields + [rank_method, ]]

    if delta_cutoff is None:
        n_li_end, n_hi_end = len(df), len(df)
    else:
        n_li_end, n_hi_end = cut_end(df[rank_method], delta_cutoff=delta_cutoff, return_n=True)

    if use_head:
        cut_length = min([n_li_end, cut_length, ])
        if cut_length < ks_sample_size:
            cut_length = ks_sample_size
        df_cut = df.head(cut_length)
    else:
        cut_length = min([n_hi_end, cut_length, ])
        if cut_length < ks_sample_size:
            cut_length = ks_sample_size
        df_cut = df.tail(cut_length)
    logger.info(f"cut ends li/hi: {n_li_end}/{n_hi_end}")
    logger.info(f"short list from ranking df {df.shape} with cut length {cut_length}")
    df_cut_ks = ks_sample_ligand_dataframe(df_cut, id_to_lig, sample_size=ks_sample_size)
    if already_taught_label is None:
        already_taught_label = []
    if len(already_taught_label) > 0:
        df_cut_ks = df_cut_ks.assign(if_taught=[lab in already_taught_label for lab in df_cut_ks['ligand_label']])
    return df_cut_ks


class OneLigandWorker(Worker):

    def __init__(
            self,
            # teach
            code_dir,
            work_dir,
            reaction_collection_json,
            prediction_ligand_pool_json,

            learner_wdir="./",
            prediction_dir="./prediction/",
            shortlist_dir="./shortlist/",

            model_path="./TwinRF_model.pkl",
            learner_json="./learner.json.gz",
            query_json=f"query_record.json.gz",
            ranking_dataframe_csv="./shortlist/qr_ranking.csv",

            complexity_percentile_cutoff=25,
            shortlist_cut_length=200,
            shortlist_sample_size=40,
            shortlist_value_cutoff=None,
            complexity_descriptor='complexity_BertzCT',
    ):
        super().__init__(name=self.__class__.__name__, code_dir=code_dir, work_dir=work_dir)
        self.shortlist_value_cutoff = shortlist_value_cutoff
        self.prediction_ligand_pool_json = prediction_ligand_pool_json
        self.shortlist_sample_size = shortlist_sample_size
        self.shortlist_cut_length = shortlist_cut_length
        self.ranking_dataframe_csv = ranking_dataframe_csv
        self.complexity_descriptor = complexity_descriptor
        self.complexity_percentile_cutoff = complexity_percentile_cutoff
        self.shortlist_dir = shortlist_dir
        self.query_json = query_json
        self.prediction_dir = prediction_dir
        self.learner_json = learner_json
        self.model_path = model_path
        self.learner_wdir = learner_wdir
        self.reaction_collection_json = reaction_collection_json

        self.complexity_cutoff = None

    @log_time
    def teach(self):
        reactions = []
        for rc_json in self.reaction_collection_json:
            rc = json_load(rc_json, gz=True)
            rc: L1XReactionCollection
            reactions += rc.reactions
        reaction_collection = L1XReactionCollection(reactions)

        learner = SingleLigandLearner.init_trfr(
            teaching_figure_of_merit='fom2',
            wdir=self.learner_wdir,
        )

        learner.teach_reactions(reaction_collection, self.model_path, tune=False, split_in_tune=True)
        json_dump(learner, self.learner_json, gz=True)

    @log_time
    def predict(self):
        ligand_pool = json_load(self.prediction_ligand_pool_json, gz=True)
        npreds = 200

        learner = json_load(self.learner_json, gz=True)
        learner: SingleLigandLearner
        ligand_amounts = learner.latest_teaching_record.reaction_collection.amount_geo_space(npreds)

        learner.load_model(-1)

        createdir(self.prediction_dir)
        chunk_size = 10
        for ichunk, lig_chunk in enumerate(tqdm(list(chunks(ligand_pool, chunk_size)))):
            prediction_chunk = learner.predict(
                lig_chunk, ligand_amounts
            )
            pkl_dump(prediction_chunk, self.prediction_dir + "/prediction_chunk_{0:06d}.pkl".format(ichunk))

    @log_time
    def query(self):
        ligands = []
        rkdfs = []
        for pkl in tqdm(sorted(glob.glob(f"{self.prediction_dir}/prediction_*.pkl"))):
            slps = pkl_load(pkl, print_timing=False)
            slps: list[SingleLigandPrediction]
            ligands += [p.ligand for p in slps]
            rkdf = SingleLigandPrediction.calculate_ranking([p.ligand for p in slps], slps, )
            rkdfs.append(rkdf)
        rkdf = pd.concat(rkdfs, axis=0, ignore_index=True)
        qr = SingleLigandPrediction.query(ligands, rkdf, self.model_path)
        json_dump(qr, self.query_json, gz=True)

    @log_time
    def shortlist(self):
        createdir(self.shortlist_dir)

        # mark already taught ligands
        reactions = []
        for rc_json in self.reaction_collection_json:
            rc = json_load(rc_json, gz=True)
            rc: L1XReactionCollection
            reactions += rc.reactions
        reaction_collection = L1XReactionCollection(reactions)
        taught_ligand_labels = [lig.label for lig in reaction_collection.ligands]

        # load query record
        qr = json_load(self.query_json, gz=True)
        qr: QueryRecord

        # calculate complexity cutoff
        comp_values = [lig.properties[self.complexity_descriptor] for lig in qr.pool]
        self.complexity_cutoff = float(np.percentile(comp_values, self.complexity_percentile_cutoff))
        logger.info("complexity cutoff: {} <= {:.4f} {:.2f}%".format(self.complexity_descriptor, self.complexity_cutoff,
                                                                     self.complexity_percentile_cutoff))
        logger.info(f"this cutoff was calculated using a pool of size: {len(qr.pool)}")

        # apply complexity cutoff to the pool and ranking dataframe
        ligand_pool = [lig for lig in qr.pool if lig.properties[self.complexity_descriptor] <= self.complexity_cutoff]
        logger.info(f"pool length after complexity cutoff: {len(ligand_pool)}")
        id_to_lig = {lig.identifier: lig for lig in ligand_pool}
        ligand_pool_inchis = [lig.identifier for lig in ligand_pool]
        Molecule.write_molecules(ligand_pool, f"{self.shortlist_dir}/ligand_pool.csv", "csv")

        ranking_dataframe = qr.ranking_dataframe
        ranking_dataframe: pd.DataFrame
        ranking_dataframe = ranking_dataframe.loc[ranking_dataframe['ligand_identifier'].isin(ligand_pool_inchis)]
        assert ligand_pool_inchis == ranking_dataframe['ligand_identifier'].tolist()
        complexity_series = pd.Series(
            [id_to_lig[i].properties[self.complexity_descriptor] for i in ranking_dataframe['ligand_identifier']]
        )
        ranking_dataframe = ranking_dataframe.assign(**{self.complexity_descriptor: complexity_series.values})
        ranking_dataframe.to_csv(self.ranking_dataframe_csv)

        # shortlists from ranking dataframe
        get_shortlist(
            ranking_dataframe=ranking_dataframe,
            rank_method="rank_average_pred_mu_top2%mu",
            cut_length=self.shortlist_cut_length,
            ks_sample_size=self.shortlist_sample_size,
            id_to_lig=id_to_lig,
            use_head=True,
            delta_cutoff=self.shortlist_value_cutoff,
            already_taught_label=taught_ligand_labels
        ).to_csv(
            f"{self.shortlist_dir}/rank_average_pred_mu_top2%mu_HEAD{self.shortlist_cut_length}_KS{self.shortlist_sample_size}.csv",
            index=False
        )

        get_shortlist(
            ranking_dataframe=ranking_dataframe,
            rank_method="rank_average_pred_mu_top2%mu",
            cut_length=self.shortlist_cut_length,
            ks_sample_size=self.shortlist_sample_size,
            id_to_lig=id_to_lig,
            use_head=False,
            delta_cutoff=self.shortlist_value_cutoff,
            already_taught_label=taught_ligand_labels
        ).to_csv(
            f"{self.shortlist_dir}/rank_average_pred_mu_top2%mu_TAIL{self.shortlist_cut_length}_KS{self.shortlist_sample_size}.csv",
            index=False
        )

        get_shortlist(
            ranking_dataframe=ranking_dataframe,
            rank_method="rank_average_pred_std",
            cut_length=self.shortlist_cut_length,
            ks_sample_size=self.shortlist_sample_size,
            id_to_lig=id_to_lig,
            use_head=True,
            delta_cutoff=self.shortlist_value_cutoff,
            already_taught_label=taught_ligand_labels
        ).to_csv(
            f"{self.shortlist_dir}/rank_average_pred_std_HEAD{self.shortlist_cut_length}_KS{self.shortlist_sample_size}.csv",
            index=False
        )

    @log_time
    def add_cas_number(self):
        inchi_field = 'ligand_identifier'
        for csv in glob.glob(f"{self.shortlist_dir}/rank_*.csv"):
            logger.info(f"working on: {csv}")
            shortlist = pd.read_csv(csv)
            if "cas_number" in shortlist.columns:
                continue
            inchis = shortlist[inchi_field].tolist()
            inchi_to_cid = ChemScraper.request_convert_identifiers(identifiers=inchis, input_type='inchi',
                                                                   output_type='cid')
            records = shortlist.to_dict(orient="records")
            for r in records:
                inchi = r[inchi_field]
                cid = inchi_to_cid[inchi]
                cas_number = ChemScraper.get_cas_number(cid)
                if cas_number is None:
                    logger.warning(f"cannot find cas number from pubchem compound, cid: {cid}")
                r.update({"cid": cid, "cas_number": cas_number, })
            shortlist = pd.DataFrame.from_records(records)
            shortlist.to_csv(csv, index=False)
