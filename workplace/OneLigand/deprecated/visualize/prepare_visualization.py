import glob
import typing
from collections import OrderedDict, defaultdict
from functools import wraps

import numpy as np
import pandas as pd
import umap
from loguru import logger
from monty.json import MSONable
from tqdm import tqdm

from lsal.alearn.one_ligand_worker import Worker, log_time, Molecule, SingleLigandPrediction
from lsal.schema import L1XReactionCollection
from lsal.utils import get_basename, get_workplace_data_folder, get_folder, file_exists, draw_svg, similarity_matrix
from lsal.utils import json_load, FilePath, calculate_distance_matrix, SEED, json_dump, pkl_load

"""
**DEPRECATED, USE ONLY MONGO BACKEND**

produce json data for visualization

not useful if using mongo backend

1. descriptor dict: data[label][des_name]<label> -> <descriptor>
2. dimred dict: 

1. molecule_dataframe: 
    1. molecule descriptors
    2. dimred results
2. molecule_predictions:
    1. a-fom predictions
    2. a-fom expt
    3. ranking params
    4. campaign meta
3. assets/*.svg: molecular structures
4. reactions: simplified L1X reactions
5. counterfactual: pool ligands
?6. counterfactual: mutation

pages:
1. molecule table: columns include `label`, `SMILES`, `InChI`, `CAS`, `structure`(svg) 
2. molecule scatter: 2d scatter plots of xy(descriptors, dimred) c(rk_params, campaign meta) with hover(expt/prediction)
"""

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)

# modify this for new iterations
_result_date = {
    "SL0519": 20220519,
    "AL1026": 20221026,
    "AL0907": 20220907,
}


class CampaignResultLocation(MSONable):
    def __init__(
            self,
            name: str,
            date: int,
            reaction_collection_json: FilePath,
    ):
        self.date = date
        self.reaction_collection_json = reaction_collection_json
        self.name = name
        assert self.name.startswith("EXPT_")
        assert file_exists(self.reaction_collection_json)

    @classmethod
    def from_name(cls, name):
        collect_folder = f"{_code_folder}/../collect"
        rc = f"{collect_folder}/reaction_collection_{name}.json.gz"
        return cls(
            name="EXPT_" + name,
            date=_result_date[name],
            reaction_collection_json=rc,
        )


class ModelResultLocation(MSONable):

    def __init__(
            self, name: str, date: int,
            reaction_collection_json: FilePath,
            ranking_df_csv: FilePath,
            vendor_csvs: dict[str, FilePath],
            prediction_pkls: list[FilePath],
    ):
        self.date = date
        self.prediction_pkls = prediction_pkls
        self.vendor_csvs = vendor_csvs
        self.ranking_df_csv = ranking_df_csv
        self.reaction_collection_json = reaction_collection_json
        self.name = name

        assert file_exists(self.ranking_df_csv)
        assert file_exists(self.reaction_collection_json)
        assert len(self.vendor_csvs) > 0
        assert len(self.prediction_pkls) > 0
        for p in self.vendor_csvs.values():
            assert file_exists(p)
        for p in self.prediction_pkls:
            assert file_exists(p)

    @classmethod
    def from_name(cls, name: str):
        learning_folder = f"{_code_folder}/../learning_{name}"
        learning_folder_work = f"{_work_folder}/../learning_{name}"
        rc = f"{learning_folder}/reaction_collection_train_{name}.json.gz"
        rkdf = f"{learning_folder}/ranking_df/qr_ranking.csv"

        vcsvs = sorted(glob.glob(f"{learning_folder}/suggestion/vendor/vendor__*.csv"))
        vendor_csvs = dict()
        for csv_path in vcsvs:
            rkparam = get_basename(csv_path).replace("vendor__", "")
            vendor_csvs[rkparam] = csv_path

        return cls(
            name=name,
            date=_result_date[name],
            reaction_collection_json=rc,
            ranking_df_csv=rkdf,
            vendor_csvs=vendor_csvs,
            prediction_pkls=sorted(glob.glob(f"{learning_folder_work}/prediction/pred*.pkl")),
        )


def smart_update(method):
    @wraps(method)
    def load_method(self):
        data_name = method.__name__.replace("load_", "")
        if data_name.startswith("df_"):
            data_filename = f"{self.work_dir}/{data_name}.csv"
        elif data_name.startswith("dict_"):
            data_filename = f"{self.work_dir}/{data_name}.json.gz"
        else:
            raise ValueError
        if file_exists(data_filename) and not self.force_update:
            logger.info(f"file exists, do not update: {data_filename}")
        else:
            logger.info(f"loading data: {data_name}")
            data = method(self)
            if data_name.startswith("df_"):
                data: pd.DataFrame
                data.to_csv(data_filename)
            elif data_name.startswith("dict_"):
                data: dict
                json_dump(data, data_filename, gz=True)
            else:
                raise ValueError

    return load_method


class L1VisExporter(Worker):
    def __init__(
            self,
            code_dir: FilePath,
            work_dir: FilePath,
            ligands_json_gz: FilePath,
            model_result_locations: list[ModelResultLocation],
            campaign_result_locations: list[CampaignResultLocation],

            # calculations based on fp similarity
            fp_type: str = 'ECFP4',
            dimred_params: typing.Tuple[str] = ('nn=50;md=0.3', 'nn=50;md=0.7'),

            # result
            result_dir: FilePath = f"{_work_folder}/result",
            force_update: bool = False,

    ):
        super().__init__(name=self.__class__.__name__, code_dir=code_dir, work_dir=work_dir)
        self.force_update = force_update
        self.campaign_result_locations = campaign_result_locations
        self.model_result_locations = model_result_locations
        self.dimred_params = dimred_params
        self.ligands_json_gz = ligands_json_gz
        self.fp_type = fp_type

        self.ligands = json_load(self.ligands_json_gz)
        self.ligands: list[Molecule]
        assert len(self.ligands) == len(set(self.ligands))

        # tmp data to save time
        self.tmp_dmat_feat_npy = f'{self.work_dir}/dmat_feat.npy'
        self.tmp_dmat_chem_npy = f'{self.work_dir}/dmat_chem.npy'
        self.tmp_slps_template = f'{self.work_dir}/slps__' + '{}' + '.pkl'

        self.result_dir = result_dir

    @log_time
    @smart_update
    def load_df_ligand(self):
        records = []
        for lig in self.ligands:
            r = {
                'LigandLabel': lig.label,
                'LigandSmiles': lig.smiles,
                'LigandIdentifier': lig.identifier,
                'LigandCas': ";".join(lig.properties['cas_number']),
            }
            records.append(r)
        return pd.DataFrame.from_records(records)

    @log_time
    @smart_update
    def load_df_descriptor(self):
        records = []
        descriptors = 'avgpol,axxpol,ayypol,azzpol,molpol,ASA+,ASA-,ASA_H,ASA_P,asa,maximalprojectionarea,maximalprojectionradius,minimalprojectionarea,minimalprojectionradius,psa,vdwsa,volume,chainatomcount,chainbondcount,fsp3,fusedringcount,rotatablebondcount,acceptorcount,accsitecount,donorcount,donsitecount,mass,hararyindex,balabanindex,hyperwienerindex,wienerindex,wienerpolarity,dipole,nHeavyAtom,nC,nN,nO,nS,nP,fragCpx,nRing,SLogP'
        descriptors = descriptors.split(",")
        for lig in self.ligands:
            assert set(lig.properties['features'].keys()) == set(descriptors)
            r = dict()
            for des in descriptors:
                r["DESCRIPTOR_" + des] = lig.properties['features'][des]
            records.append(r)
        return pd.DataFrame.from_records(records)

    @log_time
    @smart_update
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
                columns += [f"DIMRED_{space}_{nnmd}_x", f"DIMRED_{space}_{nnmd}_y", ]
            values_space = np.concatenate(values_space, axis=1)
            assert values_space.shape == (len(self.ligands), 2 * len(self.dimred_params))
            values.append(values_space)
        values = np.concatenate(values, axis=1)
        assert values.shape == (len(self.ligands), 2 * 2 * len(self.dimred_params))
        dimred_df = pd.DataFrame(values, columns=columns)
        return dimred_df

    @log_time
    @smart_update
    def load_dict_label_to_svg(self):
        label_to_svg = OrderedDict()
        for lig in self.ligands:
            smiles = lig.smiles
            label = lig.label
            svg = draw_svg(smiles, fn=None)
            label_to_svg[label] = svg
        return label_to_svg

    @log_time
    @smart_update
    def load_dict_reaction_record_table(self):
        reaction_record_dict = dict()

        rc_master = []
        for crl in self.campaign_result_locations:
            rc = json_load(crl.reaction_collection_json)
            rc_master += rc.reactions
        rc_master = L1XReactionCollection(rc_master)

        for model_result_location in self.model_result_locations:
            rc = json_load(model_result_location.reaction_collection_json)
            rc: L1XReactionCollection
            for reaction in rc.reactions:
                if reaction.identifier in reaction_record_dict:
                    continue
                else:
                    if reaction.is_reaction_real:
                        rtype = "real"
                        lig_amount = reaction.ligand_solution.amount
                        lig_label = reaction.ligand.label
                        ref_reactions = rc_master.get_reference_reactions(reaction)
                        assert len(ref_reactions) > 0
                        ref_ids = [refr.identifier for refr in ref_reactions]
                        ref_ods = [r.properties['OpticalDensity'] for r in ref_reactions]
                        ref_foms = [r.properties['FigureOfMerit'] for r in ref_reactions]
                        ref_ods_mean = np.mean(ref_ods)
                        ref_foms_mean = np.mean(ref_foms)
                        ref_ods_std = np.std(ref_ods)
                        ref_foms_std = np.std(ref_foms)
                    else:
                        lig_label = None
                        ref_ids = []
                        ref_ods = []
                        ref_foms = []
                        ref_ods_mean = None
                        ref_foms_mean = None
                        ref_ods_std = None
                        ref_foms_std = None
                        if reaction.is_reaction_nc_reference:
                            rtype = "ref"
                            lig_amount = 0.0
                        elif reaction.is_reaction_blank_reference:
                            rtype = "blank"
                            lig_amount = None
                        else:
                            raise ValueError

                    r = {
                        'ReactionIdentifier': reaction.identifier,
                        'BatchName': reaction.batch_name,
                        "LigandLabel": lig_label,
                        'OpticalDensity': reaction.properties['OpticalDensity'],
                        'FigureOfMerit': reaction.properties['FigureOfMerit'],
                        'LigandAmount': lig_amount,
                        "ReactionType": rtype,
                        'RefIds': ref_ids,
                        'RefODs': ref_ods,
                        'RefFOMs': ref_foms,
                        'RefODs_mu': ref_ods_mean,
                        'RefODs_std': ref_ods_std,
                        'RefFOMs_mu': ref_foms_mean,
                        'RefFOMs_std': ref_foms_std,
                    }
                    reaction_record_dict[reaction.identifier] = r

        return reaction_record_dict

    @log_time
    @smart_update
    def load_dict_label_to_reaction_identifiers(self):
        reaction_record_table = json_load(f"{self.work_dir}/dict_reaction_record_table.json.gz")
        lab_to_rids = defaultdict(list)
        for r in reaction_record_table.values():
            lab_to_rids[r['LigandLabel']].append(r['ReactionIdentifier'])
        return lab_to_rids

    @log_time
    @smart_update
    def load_dict_label_to_df_expt(self):
        label_to_df_expt = OrderedDict()
        reaction_record_table = json_load(f"{self.work_dir}/dict_reaction_record_table.json.gz")
        for r in reaction_record_table.values():
            lig_label = r['LigandLabel']
            if lig_label is None:
                continue
            if lig_label not in label_to_df_expt:
                label_to_df_expt[lig_label] = [r, ]
            else:
                label_to_df_expt[lig_label].append(r)

        for lab in label_to_df_expt:
            label_to_df_expt[lab].sort(key=lambda x: x['LigandAmount'])
            label_to_df_expt[lab] = pd.DataFrame.from_records(label_to_df_expt[lab])

        return label_to_df_expt

    @log_time
    @smart_update
    def load_df_meta(self):
        model_suggestions = defaultdict(list)
        possible_suggestions = []
        for ml in self.model_result_locations:
            for rkpm, csv in ml.vendor_csvs.items():
                for r in pd.read_csv(csv, low_memory=False).to_dict(orient='records'):
                    lab = r['ligand_label']
                    # suggestion = ml.name + "@@" + rkpm + "@@" + r['cluster']
                    suggestion = ml.name + "@@" + rkpm
                    model_suggestions[lab].append(suggestion)
                    possible_suggestions.append(suggestion)

        possible_suggestions = sorted(set(possible_suggestions))

        lab_to_expt_campaign = dict()
        for el in self.campaign_result_locations:
            rc = json_load(el.reaction_collection_json)
            rc: L1XReactionCollection
            for lig in rc.unique_ligands:
                lab_to_expt_campaign[lig.label] = el.name

        rs = []
        for lig in self.ligands:
            r = dict()

            # 1. does this ligand belong to the initial set?
            r['is_init'] = lig.label.startswith('LIGAND')

            # 2. in which model+rkpm is this ligand suggested?
            for s in possible_suggestions:
                r["SUGGESTION__" + s] = s in model_suggestions[lig.label]

            # 3. in which campaign is this ligand tested?
            try:
                r['ExptCampaign'] = lab_to_expt_campaign[lig.label]
            except KeyError:
                r['ExptCampaign'] = None

            rs.append(r)

        return pd.DataFrame.from_records(rs)

    @log_time
    @smart_update
    def load_df_rkp(self):
        df_rkp = pd.DataFrame.from_records([{'ligand_label': lig.label} for lig in self.ligands])
        for mrl in self.model_result_locations:
            rkdf = pd.read_csv(mrl.ranking_df_csv, low_memory=False)
            rename_cols = dict()
            for c in rkdf.columns.tolist():
                if c == 'ligand_label':
                    rename_cols[c] = c
                else:
                    rename_cols[c] = c + f"__MODEL={mrl.name}"
            rkdf.rename(columns=rename_cols, inplace=True)
            df_rkp = df_rkp.merge(rkdf, how='inner', on='ligand_label', validate="1:1")
        df_rkp = df_rkp[[c for c in df_rkp.columns.tolist() if c.startswith("rank_average_")]]
        return df_rkp

    @log_time
    @smart_update
    def load_dict_label_to_df_pred(self):
        data = dict()
        for mrl in self.model_result_locations:
            label_to_xy = dict()
            for pkl in tqdm(mrl.prediction_pkls, desc=f"loading predictions of {mrl.name}"):
                slps_chunk = pkl_load(pkl, print_timing=False)
                assert len(slps_chunk) > 0
                slps_chunk: list[SingleLigandPrediction]
                for slp in slps_chunk:
                    x = slp.amounts
                    y = slp.pred_mu
                    yerr = slp.pred_std
                    label_to_xy[slp.ligand.label] = pd.DataFrame(np.array([x, y, yerr]).T, columns=['x', 'y', 'yerr'])
            data[mrl.name] = label_to_xy
        return data

    @log_time
    @smart_update
    def load_dict_mrb_to_cfpool(self):
        with open(self.tmp_dmat_chem_npy, 'rb') as f:
            dmat_chem = np.load(f)
        ncfs = 100
        labels = [lig.label for lig in self.ligands]
        label_to_list_index = {self.ligands[i].label: i for i in range(len(self.ligands))}
        # list_index_to_label = {v: k for k, v in label_to_list_index.items()}

        # data[model_name][rank_method][base_label] -> df_cfpool
        data = defaultdict(lambda: defaultdict(dict))

        for mrl in self.model_result_locations:

            df_rkp = pd.read_csv(mrl.ranking_df_csv, low_memory=False)
            for rank_method_directed, vendor_csv in mrl.vendor_csvs.items():

                logger.info(f"calculating cfs for base ligands from: {mrl.name}: {vendor_csv}")

                df_vendor = pd.read_csv(vendor_csv, low_memory=False)
                rank_method = [c for c in df_vendor.columns if c.startswith("rank_average_")][0]

                label_to_rkp = dict(zip(
                    df_rkp['ligand_label'].tolist(), df_rkp[rank_method].tolist()
                ))

                for row in df_vendor.to_dict(orient="records"):
                    cluster = row['cluster']
                    is_taught = row['is_taught']
                    base_label = row['ligand_label']
                    base_index = label_to_list_index[base_label]
                    sim_array = dmat_chem[base_index]
                    label_to_sim = dict(zip(labels, sim_array.tolist()))
                    records_mrb = []
                    for cf_label in sorted(labels, key=lambda x: label_to_sim[x], reverse=True):
                        cf_index = label_to_list_index[cf_label]
                        if cf_index == base_index:
                            continue
                        sim = dmat_chem[base_index][cf_index]

                        if cf_label not in label_to_rkp:
                            continue

                        record = {
                            # "model_name": mrl.name,
                            # "rank_method": rank_method,
                            # "ligand_label_base": base_label,
                            "cluster_base": cluster,
                            "is_taught_base": is_taught,
                            "ligand_label_cf": cf_label,
                            "similarity": sim,
                            "rank_value_base": label_to_rkp[base_label],
                            "rank_value_cf": label_to_rkp[cf_label],
                            "rank_value_delta": label_to_rkp[base_label] - label_to_rkp[cf_label]
                        }
                        records_mrb.append(record)
                        if len(records_mrb) > ncfs:
                            break
                    df_mrb = pd.DataFrame.from_records(records_mrb)
                    data[mrl.name][rank_method][base_label] = df_mrb
        return data


if __name__ == '__main__':
    worker = L1VisExporter(
        code_dir=_code_folder,
        work_dir=_work_folder,
        ligands_json_gz=f"{_code_folder}/../../MolecularInventory/ligands.json.gz",

        model_result_locations=[
            ModelResultLocation.from_name(n) for n in ['SL0519', 'AL1026', ]
        ],
        campaign_result_locations=[
            CampaignResultLocation.from_name(n) for n in ['SL0519', 'AL1026', 'AL0907']
        ],
        fp_type="ECFP4",
        force_update=True,
    )
    worker.run(
        [
            'load_df_ligand',
            'load_df_descriptor',
            'load_df_dimred',
            'load_df_meta',
            'load_df_rkp',

            'load_dict_reaction_record_table',

            'load_dict_label_to_svg',
            'load_dict_label_to_reaction_identifiers',
            'load_dict_label_to_df_expt',
            'load_dict_label_to_df_pred',

            'load_dict_mrb_to_cfpool',
        ]
    )
