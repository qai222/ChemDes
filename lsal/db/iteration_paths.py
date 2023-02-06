import glob
import os.path

from monty.serialization import loadfn

from lsal.utils import MSONable, FilePath, get_basename, file_exists, get_folder, get_workplace_data_folder

_code_folder = get_folder(__file__)


class IterationPaths(MSONable):
    def __init__(self, name: str, expt_rc_json: FilePath, round_index: int, model_folder: FilePath = None,
                 is_extra=False):
        self.is_extra = is_extra
        self.round_index = round_index
        self._model_folder = model_folder
        self._expt_rc_json = expt_rc_json
        self.name = name

    def __repr__(self):
        return self.as_dict().__repr__()

    @property
    def path_pred_folder(self):
        if self.model_folder is not None:
            wdir = get_workplace_data_folder(self.model_folder + "/__init__.py")
            return os.path.join(wdir, "prediction")

    @property
    def model_folder(self):
        try:
            return os.path.abspath(self._model_folder)
        except TypeError:
            return None

    @property
    def expt_rc_json(self):
        return os.path.abspath(self._expt_rc_json)

    @property
    def path_ranking_dataframe(self) -> FilePath:
        if self.model_folder is not None:
            return os.path.join(self.model_folder, "ranking_df/qr_ranking.csv")

    @property
    def path_vendor_folder(self) -> FilePath:
        if self.model_folder is not None:
            return os.path.join(self.model_folder, "suggestion/vendor/")

    @property
    def path_training_rc_json(self) -> FilePath:
        if self.model_folder is not None:
            return os.path.join(self.model_folder, f"reaction_collection_train_{self.name}.json.gz")

    @property
    def path_dict_vendor(self) -> dict[str, FilePath]:
        if self.model_folder is not None:
            data = dict()
            for csv in glob.glob(self.path_vendor_folder + "/vendor*.csv"):
                _, u_score, space, direction = get_basename(csv).split("__")
                data[f"{u_score} @ {direction}"] = csv
            return data

    def validate(self):
        if not file_exists(self.expt_rc_json):
            return False
        if not file_exists(self.path_ranking_dataframe):
            return False
        if not len(self.path_dict_vendor) > 0:
            return False
        if not file_exists(self.path_training_rc_json):
            return False
        return True


def load_cps(yaml_file: FilePath = f"{_code_folder}/campaign_paths.yaml"):
    cps = []
    for d in loadfn(yaml_file):
        cp = IterationPaths.from_dict(d)
        cps.append(cp)
    return sorted(cps, key=lambda x: x.round_index)
