import logging
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor

from lsal.campaign.loader import LoaderLigandDescriptors, Molecule
from lsal.campaign.single_learner import SingleLigandLearner, ReactionCollection
from lsal.twinsk.estimator import TwinRegressor
from lsal.utils import SEED, json_load, json_dump


def get_field_plfom(columns: list[str]):
    fom_col = [c for c in columns if c.strip("'").endswith("PL_FOM")][0]
    return fom_col


def get_field_sumod(columns: list[str]):
    fom_col = [c for c in columns if c.strip("'").endswith("PL_sum/OD390")][0]
    return fom_col


experiment_input_files = [
    "data/2022_0519_SL01_0008_0015_robotinput.xls",
    "data/2022_0519_SL02_0007_0012_robotinput.xls",
    "data/2022_0520_SL03_0017_0004_robotinput.xls",
    "data/2022_0520_SL04_0006_0009_robotinput.xls",
    "data/2022_0520_SL05_0018_0014_robotinput.xls",
    "data/2022_0520_SL06_0003_0010_robotinput.xls",
    "data/2022_0520_SL07_0013_0021_robotinput.xls",
    "data/2022_0520_SL08_0023_0000_robotinput.xls",
    "data/2022_0525_SL09_0000_0001_robotinput.xls",
    "data/2022_0525_SL10_0020_0002_robotinput.xls",
    "data/2022_0525_SL11_0005_0022_robotinput.xls",
]

experiment_output_files = [
    "data/PS0519_SL01_peakInfo.csv",
    "data/PS0519_SL02_peakInfo.csv",
    "data/PS0520_SL03_peakInfo.csv",
    "data/PS0520_SL04_peakInfo.csv",
    "data/PS0520_SL05_peakInfo.csv",
    "data/PS0520_SL06_peakInfo.csv",
    "data/PS0520_SL07_peakInfo.csv",
    "data/PS0520_SL08_peakInfo.csv",
    "data/PS0525_SL09_peakInfo.csv",
    "data/PS0525_SL10_peakInfo.csv",
    "data/PS0525_SL11_peakInfo.csv",
]


def spline_fit_single_ligand_campaign(reactions: ReactionCollection, get_fom):
    l2rs = {lc[0]: rs for lc, rs in reactions.get_lcomb_to_reactions().items()}
    for l, rs in l2rs.items():
        x = [r.ligand_solutions[0].amount for r in rs]
        y = [r.properties[get_fom(r.properties)] for r in rs]
        y = np.nan_to_num(y, nan=0.0)
        # TODO we assumed reaction in rs only differ in ligand amount
        y = [yy for _, yy in sorted(zip(x, y), key=lambda t: t[0])]
        x = sorted(x)
        if len(x) > len(set(x)):
            unique_xs = []
            for xx in x:
                while xx in unique_xs:
                    xx += 1e-6
                unique_xs.append(xx)
            x = unique_xs
        # cs = CubicSpline(x, y, bc_type="natural")
        cs = UnivariateSpline(np.log(x), y)
        fitted_y = cs(np.log(x))
        plt.scatter(np.log(x), y)
        plt.plot(np.log(x), fitted_y)
        plt.savefig("fit/{}.png".format(l.label))
        plt.clf()
        for r, xx, yy in zip(rs, x, fitted_y):
            r.ligand_solutions[0].volume = 1
            r.ligand_solutions[0].concentration = xx
            r.properties[get_fom(r.properties)] = yy
    return reactions


def smooth(x, y):
    xx = np.linspace(min(x), max(x), 1000)

    # interpolate + smooth
    itp = interp1d(x, y, kind='linear')
    window_size, poly_order = 801, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    return xx, yy_sg


def fit_single_ligand_campaign(reactions: ReactionCollection, get_fom):
    l2rs = {lc[0]: rs for lc, rs in reactions.get_lcomb_to_reactions().items()}
    for l, rs in l2rs.items():
        x = [r.ligand_solutions[0].amount for r in rs]
        y = [r.properties[get_fom(r.properties)] for r in rs]
        y = np.nan_to_num(y, nan=0.0)
        # # TODO we assumed reaction in rs only differ in ligand amount
        y = [yy for _, yy in sorted(zip(x, y), key=lambda t: t[0])]
        x = sorted(x)
        if len(x) > len(set(x)):
            unique_xs = []
            for xx in x:
                while xx in unique_xs:
                    xx += 1e-6
                unique_xs.append(xx)
            x = unique_xs
        fitted_x, fitted_y = smooth(x, y)
        plt.plot(fitted_x, fitted_y)
        plt.scatter(x, y)
        plt.savefig("{}.png".format(l.label))
        plt.clf()
        for r, xx, yy in zip(rs, fitted_x, fitted_y):
            r.ligand_solutions[0].volume = 1
            r.ligand_solutions[0].concentration = xx
            r.properties[get_fom(r.properties)] = yy
    return reactions


class SingleWorkflow:
    def __init__(
            self, ligand_inventory_file, solvent_inventory_file, ligand_descriptors_file,
            experiment_input_files,
            experiment_output_files,
            get_fom_field=get_field_plfom,
            prefit=False,
            n_predictions=200,
            exclude_reaction_function=None,
    ):
        self.prefit = prefit
        self.n_predictions = n_predictions
        self.get_fom_field = get_fom_field
        self.ligand_inventory = json_load(ligand_inventory_file)
        self.solvent_inventory = json_load(solvent_inventory_file)
        self.ligand_to_des_record = LoaderLigandDescriptors("test").load(ligand_descriptors_file, self.ligand_inventory)
        self.learner = SingleLigandLearner(
            [], [], TwinRegressor(RandomForestRegressor(n_estimators=10, random_state=SEED)),
            self.get_fom_field,
            self.ligand_to_des_record
        )
        collected_reactions = ReactionCollection.from_files(
            experiment_input_files=experiment_input_files,
            experiment_output_files=experiment_output_files,
            ligand_inventory=self.ligand_inventory,
            solvent_inventory=self.solvent_inventory,
        )
        if exclude_reaction_function is None:
            self.campaign_reactions = collected_reactions
        else:
            self.campaign_reactions = ReactionCollection(
                [r for r in collected_reactions.reactions if not exclude_reaction_function(r)],
                collected_reactions.properties,
        )

        # map ref to each reaction
        for r in self.campaign_reactions.real_reactions:
            ref_reactions = []
            for ref_r in self.campaign_reactions.ref_reactions:
                if ref_r.identifier.split("@@")[0] == r.identifier.split("@@")[0]:
                    ref_reactions.append(ref_r.identifier)
            r.properties["ref_reaction_identifiers"] = ref_reactions

        # spline fit for campaign reactions
        if prefit:
            self.campaign_reactions = spline_fit_single_ligand_campaign(self.campaign_reactions, self.get_fom_field)
        # self.campaign_reactions = fit_single_ligand_campaign(self.campaign_reactions, self.get_fom_field)

        self.unique_ligands = [lc[0] for lc in self.campaign_reactions.unique_lcombs]
        self.ligand_to_reactions = {lc[0]: reactions for lc, reactions in
                                    self.campaign_reactions.get_lcomb_to_reactions().items()}

        self.uncertainty = None
        self.mae = None
        self.average_uncertainty = None
        self.average_mae = None
        self.suggestions = None

        self.amount_min, self.amount_max, self.amount_unit = self.campaign_reactions.ligand_amount_range
        # self.amounts = np.linspace(self.amount_min, self.amount_max, self.n_predictions)
        self.amounts = np.geomspace(self.amount_min, self.amount_max, self.n_predictions)

        self.history = OrderedDict()

    @property
    def unlearned_ligands(self):
        return [lig for lig in self.unique_ligands if lig not in self.learner.learned_ligands]

    @property
    def unlearned_reactions(self):
        return [r for r in self.campaign_reactions.real_reactions if r not in self.learner.learned_reactions]

    def teach_ligands(self, ligands: list[Molecule]):
        reactions_to_teach = ReactionCollection.subset_by_lcombs(self.campaign_reactions, [(ligand,) for ligand in
                                                                                           self.learner.learned_ligands + ligands])
        logging.warning("TEACHING...")
        self.learner.teach(reactions_to_teach)
        logging.warning("AFTER TEACHING")
        logging.warning(self.learner.status)
        logging.warning("reactions learned vs unlearned: {} vs {}".format(len(self.learner.learned_reactions),
                                                                          len(self.unlearned_reactions)))
        if len(self.unlearned_reactions) == 0:
            return
        self.uncertainty = self.learner.eval_pred_uncertainty(self.unlearned_ligands, amounts=self.amounts)
        self.mae = self.learner.eval_pred_wrt_real(ReactionCollection(self.unlearned_reactions))
        self.average_uncertainty = np.mean(list(self.uncertainty.values()))
        self.average_mae = np.mean(list(self.mae.values()))
        self.largest_mae = max(self.mae.values())
        self.largest_uncertainty = max(self.uncertainty.values())
        logging.warning("average uncertainty for unlearned: {:.4f}".format(self.average_uncertainty))
        logging.warning("average mae for unlearned: {:.4f}".format(self.average_mae))
        self.suggestions = self.learner.suggest_ligand(
            self.unlearned_ligands, self.n_predictions, self.amount_min, self.amount_max, "large_average_std", k=1
        )

    def train_seed(self, size=2, seed=SEED):
        random.seed(seed)
        seed_ligands = random.sample(self.unique_ligands, size)
        logging.warning("seed ligands: {}".format(seed_ligands))
        self.teach_ligands(seed_ligands)

    def teach(self, nligands: int = None, init_size: int = 2, init_seed: int = SEED):
        if nligands is None:
            nligands = len(self.unique_ligands)
        nligands_taught = 0
        logging.warning("learn # of ligands: {}".format(nligands))
        self.train_seed(init_size, init_seed)
        nligands_taught += init_size
        while nligands_taught < nligands:
            nsuggested = len(self.suggestions)
            self.teach_ligands(self.suggestions)
            nligands_taught += nsuggested
            self.history[nligands_taught] = {
                "average_uncertainty": self.average_uncertainty,
                "average_mae": self.average_mae,
                "largest_mae": self.largest_mae,
                "largest_uncertainty": self.largest_uncertainty,
            }
            visdata = self.visualize_iteration()
            json_dump(visdata, "vis/vis-{0:0>4}.json".format(nligands_taught))

    def get_real_xy(self):
        data = dict()
        for ligand, reactions in self.ligand_to_reactions.items():
            real_xs = np.array([r.ligand_solutions[0].amount for r in reactions])
            real_ys = [r.properties[self.get_fom_field(r.properties.keys())] for r in reactions]
            data[ligand.smiles] = real_xs, real_ys
        return data

    def visualize_iteration(self):
        ligand_label_to_vis_data = dict()

        uncertainty = self.learner.eval_pred_uncertainty(self.unique_ligands, amounts=self.amounts)
        mae = self.learner.eval_pred_wrt_real(self.campaign_reactions)

        for ligand, reactions in self.ligand_to_reactions.items():
            fake_xs = self.amounts
            real_ys = [r.properties[self.get_fom_field(r.properties.keys())] for r in reactions],
            llp = self.learner.predict([ligand, ], fake_xs)[0]
            ligand_label_to_vis_data[ligand.label] = dict(
                real_xs=np.array([r.ligand_solutions[0].amount for r in reactions]),
                real_ys=np.nan_to_num(np.array(real_ys))[0],
                is_learned=ligand in self.learner.learned_ligands,
                is_suggested=ligand in self.suggestions,
                fake_xs=fake_xs,
                fake_ys=llp.pred_mu,
                fake_ys_err=llp.pred_std,
                mae=mae[ligand],
                uncertainty=uncertainty[ligand],
                history=self.history,
            )
        return ligand_label_to_vis_data


