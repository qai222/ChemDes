import inspect
import logging
import random
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np

from lsal.campaign.loader import Molecule
from lsal.campaign.single_learner import SingleLigandLearner, ReactionCollection, SingleLigandPredictions
from lsal.utils import SEED, get_timestamp


class SingleWorkflow:
    def __init__(
            self,
            reaction_collection: ReactionCollection,
            learner: SingleLigandLearner,
            n_predictions=200,  # num of predictions generated for each ligand over the concentration range
            learner_history: OrderedDict = None,
            reg_tune=False,
            amount_space: str = "geo",
            timing:dict=None,
            # TODO add spline fitting
    ):
        if timing is None:
            timing = dict()
        self.timing = timing
        self.amount_space = amount_space
        self.learner = learner
        self.reg_tune = reg_tune
        self.fom_field = self.learner.fom_def
        self.ligand_to_des_record = self.learner.ligand_to_des_record
        self.n_predictions = n_predictions
        self.campaign_reactions = reaction_collection  # this should not be changed after init
        if learner_history is None:
            learner_history = OrderedDict()
        self.learner_history = learner_history
        self.amount_min, self.amount_max, self.amount_unit = self.campaign_reactions.ligand_amount_range

    @property
    def amounts(self):
        if self.amount_space == "geo":
            return self.amount_geo_space
        elif self.amount_space == "lin":
            return self.amount_lin_space
        else:
            raise NotImplementedError("unknown space: {}".format(self.amount_space))

    @property
    def unique_ligands(self):
        return [lc[0] for lc in self.campaign_reactions.unique_lcombs]

    @property
    def ligand_to_reactions(self):
        return {lc[0]: reactions for lc, reactions in self.campaign_reactions.get_lcomb_to_reactions().items()}

    @property
    def amount_lin_space(self):
        return np.linspace(self.amount_min, self.amount_max, self.n_predictions)

    @property
    def amount_geo_space(self):
        return np.geomspace(self.amount_min, self.amount_max, self.n_predictions)

    @property
    def unlearned_ligands(self):
        return [lig for lig in self.unique_ligands if lig not in self.learner.learned_ligands]

    @property
    def unlearned_reactions(self):
        return [r for r in self.campaign_reactions.real_reactions if r not in self.learner.learned_reactions]

    @property
    def wf_status(self) -> dict[str, Any]:
        data = dict()
        # learned info
        data["learned_ligands"] = deepcopy(self.learner.learned_ligands)
        # data["learned_reactions"] = deepcopy(self.learner.learned_reactions)  # not really useful...

        # single ligand predictions
        slpreds = self.learner.predict(self.unique_ligands, self.amounts, self.ligand_to_des_record)
        data["predictions"] = slpreds

        # ranking
        data["rank_data"] = SingleLigandPredictions.rank_ligands(slpreds)

        # compare wrt real
        data["mae_wrt_real"] = self.learner.eval_pred_wrt_real(self.campaign_reactions.real_reactions)
        return data

    def teach_ligands(self, ligands: list[Molecule]):
        """
        *update* the learner with the reactions (from campaign reactions) of the given ligands

        :param ligands: a list of ligands, if a ligand is already learned, it's excluded
        :param rank_unlearned: if write sorted unlearned ligands into history
        :return:
        """
        teaching_list = [lig for lig in ligands if lig not in self.learner.learned_ligands]
        reactions_to_teach = ReactionCollection.subset_by_lcombs(
            self.campaign_reactions, [(ligand,) for ligand in self.learner.learned_ligands + teaching_list]
        )
        logging.warning("TEACHING...")
        ts1 = time.perf_counter()
        self.learner.teach(reactions_to_teach.reactions, tune=self.reg_tune)
        ts2 = time.perf_counter()
        logging.warning("AFTER TEACHING")
        logging.warning(self.learner.status)
        logging.warning("reactions learned vs unlearned: {} vs {}".format(len(self.learner.learned_reactions),
                                                                          len(self.unlearned_reactions)))
        this_round = get_timestamp()
        self.learner_history.update(
            {"{}: {}".format(inspect.stack()[0].function, this_round): self.wf_status}
        )
        self.timing[this_round] = ts2 - ts1

    def train_seed(self, size=2, seed=SEED):
        assert size <= len(self.unique_ligands)
        seed_ligands = random.Random(seed).sample(self.unique_ligands, size)
        logging.warning("seed ligands: {}".format(seed_ligands))
        self.teach_ligands(seed_ligands)

    def teach_one_by_one(
            self, metric: str, num_ligands_to_teach: int = None, init_size: int = 2, init_seed: int = SEED
    ):
        if num_ligands_to_teach is None:
            num_ligands_to_teach = len(self.unique_ligands)
        num_ligands_taught = 0
        logging.warning("learn # of ligands: {}".format(num_ligands_to_teach))
        self.train_seed(init_size, init_seed)
        num_ligands_taught += init_size
        while num_ligands_taught < num_ligands_to_teach:
            last_history_key, last_history_status = next(reversed(self.learner_history.items()))
            last_rank_data = last_history_status["rank_data"][metric]
            last_rank_data = [lv for lv in last_rank_data if lv[0] in self.unlearned_ligands]
            suggested_ligand, suggested_value = last_rank_data[0]
            logging.warning("learning suggestion: {} with value {:.4f}".format(suggested_ligand, suggested_value))
            self.teach_ligands([suggested_ligand, ])
            num_ligands_taught += 1

    def visualize_history_data(self, status_data: dict):
        learned_ligands = status_data["learned_ligands"]
        # learned_reactions = status_data["learned_reactions"]
        rank_data = status_data["rank_data"]
        mae_wrt_real = status_data["mae_wrt_real"]
        sllps = status_data["predictions"]
        visualization_data = dict()

        for sllp in sllps:
            sllp: SingleLigandPredictions
            fake_xs = self.amounts
            ligand = sllp.ligand
            real_reactions = self.ligand_to_reactions[ligand]
            real_ys = [r.properties[self.fom_field] for r in real_reactions]
            real_xs = [r.ligand_solutions[0].amount for r in real_reactions]
            visualization_data[ligand] = dict(
                real_xs=real_xs,
                real_ys=real_ys,
                is_learned=ligand in learned_ligands,
                fake_xs=fake_xs,
                fake_ys=sllp.pred_mu,
                fake_ys_err=sllp.pred_std,
                uncertainty=sllp.overall_uncertainty(),
                uncertainty_top2=sllp.overall_uncertainty(0.02),
                mae_wrt_real=mae_wrt_real[ligand],
            )
            for m in rank_data:
                visualization_data[ligand]["rank-" + m] = [rd[0] for rd in rank_data[m]].index(ligand)
        return visualization_data
