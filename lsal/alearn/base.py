import abc
from collections import OrderedDict
from datetime import datetime

import pandas as pd
from loguru import logger
from monty.json import MSONable

from lsal.utils import FilePath
from lsal.utils import pkl_load, msonable_repr


class ActiveLearningRecord(MSONable, abc.ABC):
    def __init__(
            self, date: datetime,
            properties: OrderedDict = None,
    ):
        self.date = date
        if properties is None:
            properties = OrderedDict()
        self.properties = properties

    def __repr__(self):
        return msonable_repr(self)

    def __hash__(self):
        return hash(self.date)

    def __gt__(self, other):
        return self.date.__gt__(other.date)

    def __lt__(self, other):
        return self.date.__lt__(other.date)

    def __eq__(self, other):
        return self.date == other.date


class TeachingRecord(ActiveLearningRecord):
    def __init__(self, date: datetime,
                 model_path: FilePath,
                 X, y,
                 properties: OrderedDict = None):
        """
        record a teaching event of an active learner
        each associated to a finished fit and a saved model

        :param model_path: where is the latest model located
        """
        super().__init__(date, properties)
        self.y = y
        self.X = X
        self.model_path = model_path
        self.date = date


class MetaLearner(MSONable, abc.ABC):
    def __init__(
            self,
            work_dir: FilePath,
            teaching_figure_of_merit: str,
            teaching_records: list[TeachingRecord] = None,
    ):
        """
        active learner for single ligand reactions
        """
        self.work_dir = work_dir
        self.teaching_figure_of_merit = teaching_figure_of_merit
        if teaching_records is None:
            teaching_records = []
        self.teaching_records = teaching_records

        self.current_model = None

    @property
    def model_paths(self) -> list[FilePath]:
        return [tr.model_path for tr in self.teaching_records]

    def load_model(self, model_index=-1):
        assert len(self.teaching_records) > 0
        if model_index == -1:
            model_index = len(self.teaching_records) - 1
        mpath = self.model_paths[model_index]
        logger.info(f"loading the ** {model_index} **th model of {self.__class__.__name__}")
        self.current_model = pkl_load(mpath)

    @classmethod
    @abc.abstractmethod
    def init_trfr(
            cls,
            teaching_figure_of_merit: str,
            wdir: FilePath,
    ):
        pass

    @abc.abstractmethod
    def teach_reactions(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass


class QueryRecord(ActiveLearningRecord):
    def __init__(self, date: datetime, predictor_path: FilePath, pool: list,
                 ranking_dataframe: pd.DataFrame, query_results: dict, properties: OrderedDict = None):
        """
        record a query event of an active learner
        """
        super().__init__(date, properties)
        self.pool = pool
        self.predictor_path = predictor_path
        self.query_results = query_results
        self.ranking_dataframe = ranking_dataframe
