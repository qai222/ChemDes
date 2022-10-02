import base64
import collections
import functools
import gzip
import inspect
import itertools
import json
import logging
import os
import pathlib
import pickle
import re
import shutil
import sys
import time
import typing
from datetime import datetime
from io import BytesIO
from itertools import zip_longest
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import monty.json
import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger
from monty.json import MSONable
from rdkit.Chem import MolToSmiles, MolToInchi, MolFromSmiles, MolFromSmarts
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.inchi import MolFromInchi
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

SEED = 42

FilePath = typing.Union[pathlib.Path, os.PathLike, str]
Xtype = typing.Union[np.ndarray, list[list[float]]]
ytype = typing.Union[np.ndarray, list[float]]


def inspect_tasks(task_header='task_') -> dict[str, typing.Callable]:
    # https://stackoverflow.com/questions/139180/
    return {f[0].replace(f'{task_header}', ''): f[1] for f in
            inspect.getmembers(sys.modules['__main__'], inspect.isfunction) if
            f[0].startswith(f'{task_header}')}


def get_workplace_data_folder(f: str) -> FilePath:
    absf = os.path.abspath(f)
    assert 'workplace' in absf and absf.endswith('.py') and 'workplace_data' not in absf
    absf = absf.replace('workplace', 'workplace_data', 1)
    folder = get_folder(absf)
    createdir(folder)
    return folder


def msonable_repr(msonable: MSONable, precision=5):
    s = "{}: ".format(msonable.__class__.__name__)
    for k, v in msonable.as_dict().items():
        if k.startswith("@"):
            continue
        if isinstance(v, float):
            v = round(v, precision)
        s += "{}={}\t".format(k, v)
    return s


def file_exists(fn: FilePath):
    return os.path.isfile(fn) and os.path.getsize(fn) > 0


def inchi2smiles(inchi: str) -> str:
    return MolToSmiles(MolFromInchi(inchi))


def smiles2inchi(smi: str) -> str:
    return MolToInchi(MolFromSmiles(smi))


def neutralize_atoms(mol):
    pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def to_float(x):
    try:
        assert not np.isnan(x)
        return float(x)
    except (ValueError, AssertionError) as e:
        return None


def json_dump(o, fn: FilePath, gz=False):
    if gz:
        with gzip.open(fn, 'wt') as f:
            json.dump(o, f, cls=monty.json.MontyEncoder)
    else:
        with open(fn, "w") as f:
            json.dump(o, f, cls=monty.json.MontyEncoder)


def json_load(fn: FilePath, warning=False, gz=False):
    if warning:
        logging.warning("loading file: {}".format(fn))
    if gz:
        with gzip.open(fn, 'rt') as f:
            o = json.load(f, cls=monty.json.MontyDecoder)
    else:
        with open(fn, "r") as f:
            o = json.load(f, cls=monty.json.MontyDecoder)
    return o


# https://stackoverflow.com/questions/31174295
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def strip_extension(p: FilePath):
    return os.path.splitext(p)[0]


def unison_shuffle(a, b, seed):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=seed).permutation(len(a))
    return a[p], b[p]


def sort_and_group(data, keyf):
    groups = []
    unique_keys = []
    data = sorted(data, key=keyf)
    for k, g in itertools.groupby(data, keyf):
        groups.append(list(g))
        unique_keys.append(k)
    return unique_keys, groups


def get_folder(path: typing.Union[pathlib.Path, str]):
    return os.path.dirname(os.path.abspath(path))


def padding_vial_label(v: str) -> str:
    """ pad zeros for vial label, e.g. A1 -> A01 """
    assert len(v) <= 3
    v_list = re.findall(r"[^\W\d_]+|\d+", v)
    try:
        assert len(v_list) == 2
        alpha, numeric = v_list
        assert len(numeric) <= 2
        final_string = alpha + "{0:02d}".format(int(numeric))
        assert len(final_string) == len(alpha) + 2
        return final_string
    except (AssertionError, ValueError) as e:
        raise ValueError("invalid vial: {}".format(v))


def get_basename(path: FilePath):
    return strip_extension(os.path.basename(os.path.abspath(path)))


def get_extension(path: FilePath):
    return os.path.splitext(path)[-1][1:]


def scale_df(data_df: pd.DataFrame):
    data_df = data_df.select_dtypes('number')
    x = data_df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_df = pd.DataFrame(x_scaled, columns=data_df.columns, index=data_df.index)
    return data_df


def truncate_distribution(x: list[float] or np.ndarray, position="top", fraction=0.1, return_indices=False):
    """ keep top or bottom x% of the population """
    if isinstance(x, list):
        x = np.array(x)
    assert x.ndim == 1
    nsize = int(len(x) * fraction)
    if nsize == 0:
        nsize = 1
    if position == "top":
        indices = np.argpartition(x, -nsize)[-nsize:]
    else:
        indices = np.argpartition(x, nsize)[:nsize]
    if return_indices:
        return indices
    else:
        return x[indices]


def unique_element_to_indices(ligands):
    unique_ligand_to_indices = dict()
    for iligand, ligand in enumerate(ligands):
        if ligand not in unique_ligand_to_indices:
            unique_ligand_to_indices[ligand] = [iligand, ]
        else:
            unique_ligand_to_indices[ligand].append(iligand)
    return unique_ligand_to_indices


def pkl_dump(o, fn: FilePath, print_timing=True) -> None:
    ts1 = time.perf_counter()
    with open(fn, "wb") as f:
        pickle.dump(o, f)
    ts2 = time.perf_counter()
    if print_timing:
        logger.info("dumped {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))


def pkl_load(fn: FilePath, print_timing=True):
    ts1 = time.perf_counter()
    with open(fn, "rb") as f:
        d = pickle.load(f)
    ts2 = time.perf_counter()
    if print_timing:
        logger.info("loaded {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))
    return d


def read_smi(smifile: FilePath):
    with open(smifile, "r") as f:
        lines = f.readlines()
    return [smi.strip() for smi in lines if len(smi.strip()) > 0]


def write_smi(smis: list[str], outfile: FilePath):
    with open(outfile, "w") as f:
        for smi in smis:
            f.write(smi + "\n")


def remove_stereo(smi: str):
    smi = smi.replace("/", "").replace("\\", "").replace("@", "").replace("@@", "")
    return smi


def parse_formula(formula: str) -> dict[str, float]:  # from pymatgen
    def get_sym_dict(form: str, factor) -> dict[str, float]:
        sym_dict: dict[str, float] = collections.defaultdict(float)
        for m in re.finditer(r"([A-Z][a-z]*)\s*([-*\.e\d]*)", form):
            el = m.group(1)
            amt = 1.0
            if m.group(2).strip() != "":
                amt = float(m.group(2))
            sym_dict[el] += amt * factor
            form = form.replace(m.group(), "", 1)
        if form.strip():
            raise ValueError(f"{form} is an invalid formula!")
        return sym_dict

    m = re.search(r"\(([^\(\)]+)\)\s*([\.e\d]*)", formula)
    if m:
        factor = 1.0
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join([f"{el}{amt}" for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    return get_sym_dict(formula, 1)


def createdir(directory):
    """
    mkdir
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_timestamp():
    return int(datetime.now().timestamp() * 1000)


def removefile(what: FilePath):
    try:
        os.remove(what)
    except OSError:
        pass


def removefolder(what: FilePath):
    try:
        os.rmdir(what)
    except OSError:
        pass


def is_close(a: float, b: float, eps=1e-5):
    return abs(a - b) < eps


def is_close_relative(a: float, b: float, eps=1e-5):
    aa = abs(a)
    bb = abs(b)
    try:
        return abs(a - b) / min([aa, bb]) < eps
    except ZeroDivisionError:
        try:
            return abs(a - b) / max([aa, bb]) < eps
        except ZeroDivisionError:
            return True


def is_close_list(lst: list[float], eps=1e-5):
    for i, j in itertools.combinations(lst, 2):
        if is_close(i, j, eps):
            return True
    return False


def passmein(func):
    """
    https://stackoverflow.com/questions/8822701
    """

    def wrapper(*args, **kwargs):
        return func(func, *args, **kwargs)

    return wrapper


def docstring_parameter(*sub):
    """ https://stackoverflow.com/questions/10307696 """

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def smi2imagestr(smi: str):
    m = MolFromSmiles(smi)
    img = MolToImage(m)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue())
    src_str = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return src_str


def plot_molcloud(smis: list[str], width: float, outfile: FilePath):
    import matplotlib.pyplot as plt
    import molcloud
    bgc = "#f5f4e9"
    node_size = 100
    plt.figure(figsize=(width, width))
    molcloud.plot_molcloud(smis, background_color=bgc, node_size=node_size)
    plt.savefig(outfile, dpi=600)


def split_file(
        filename: FilePath, n=1000, outfile_template="outfile_{0:03d}.out",
):
    """
    split a large file into smaller chunks
    """

    def grouper(n, iterable, fillvalue=None):
        """Collect data into fixed-length chunks or blocks"""
        # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    with open(filename) as f:
        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            with open(outfile_template.format(i), 'w') as fout:
                fout.writelines(g)


def combine_files(filenames: list[FilePath], outfile: FilePath):
    c = ""
    for filename in filenames:
        with open(filename, "r") as f:
            c += f.read()
    with open(outfile, "w") as f:
        f.write(c)


def has_isotope(mol):
    """ if any atom is an isotope """
    if isinstance(mol, str):
        mol = MolFromSmiles(mol)
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            return True
    return False


def upper_confidence_interval(data: np.ndarray, confidence=0.95):
    # TODO ubc algorithm uses sample size to penalize mean, we have a fixed plate, sample size stays the same
    """ https://stackoverflow.com/questions/15033511/ """
    assert data.ndim == 1 and len(data) >= 2
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m + h


def log_time(method):
    def timed(*args, **kw):
        logger.info(f"WORKING ON: {method.__name__}")
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info("TIME COST: {:.3f} s".format(te - ts))
        return result

    return timed


def get_date_ymd() -> str:
    return datetime.now().strftime("%Y_%m_%d")


def calculate_distance_matrix(descriptor_dataframe: pd.DataFrame, metric="manhattan", scale=True):
    logger.warning(f"descriptor_dataframe shape: {descriptor_dataframe.shape}")
    descriptor_dataframe = descriptor_dataframe.select_dtypes('number')
    logger.warning(f"descriptor_dataframe shape numbers only: {descriptor_dataframe.shape}")
    if scale:
        logger.warning("the columns of input dataframe are scaled to [0, 1] for distance matrix")
        df = scale_df(descriptor_dataframe)
    else:
        df = descriptor_dataframe
    distance_matrix = pairwise_distances(df.values, metric=metric)
    return distance_matrix


def get_file_size(fn: FilePath, unit="m"):
    """
    see https://stackoverflow.com/questions/6080477
    improvement https://code.activestate.com/recipes/577081/
    """
    assert unit in ('b', 'k', 'm', 'g')
    bsize = os.path.getsize(fn)
    if unit == 'b':
        return bsize
    elif unit == 'k':
        return bsize / 1024
    elif unit == 'm':
        return bsize / (1024 ** 2)
    elif unit == 'g':
        return bsize / (1024 ** 3)


def size_report(o):
    size = sys.getsizeof(o)
    # size = dict()
    # d = o.as_dict()
    # for k, v in d.items():
    #     s = sys.getsizeof(v)
    #     s = f"{round(s / (1024 * 1024), 4)} MB"
    #     size[k] = s
    return size


def flatten_json(nested_json: dict, exclude: list = [''], sep: str = '___') -> dict:
    """
    directly taken from https://stackoverflow.com/questions/58442723
    see also https://stackoverflow.com/questions/52795561
    Flatten a list of nested dicts.
    """
    out = dict()

    def flatten(x: (list, dict, str), name: str = '', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude:
                    flatten(x[a], f'{name}{a}{sep}')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, f'{name}{i}{sep}')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def movefile(what, where):
    """
    shutil operation to move
    :param what:
    :param where:
    :return:
    """
    try:
        shutil.move(what, where)
    except IOError:
        os.chmod(where, 777)
        shutil.move(what, where)


def copyfile(what, where):
    """
    shutil operation to copy
    :param what:
    :param where:
    :return:
    """
    try:
        shutil.copy(what, where)
    except IOError:
        os.chmod(where, 777)
        shutil.copy(what, where)


def is_sorted_ascend(sample):
    return all(sample[i] <= sample[i + 1] for i in range(len(sample) - 1))


def is_sorted_descend(sample):
    return all(sample[i] >= sample[i + 1] for i in range(len(sample) - 1))


def cut_end(sample, delta_cutoff=0.05, return_n=False):
    actual_delta = abs(sample[0] - sample[-1])
    actual_delta_cutoff = delta_cutoff * actual_delta
    assert is_sorted_ascend(sample) or is_sorted_descend(sample)
    li_end = []
    hi_end = []
    i = 0
    while abs(sample[i] - sample[0]) <= actual_delta_cutoff:
        li_end.append(sample[i])
        i += 1
    i = len(sample) - 1
    while abs(sample[i] - sample[-1]) <= actual_delta_cutoff:
        hi_end.append(sample[i])
        i -= 1
    if return_n:
        return len(li_end), len(hi_end)
    else:
        return li_end, hi_end


def download_file(url: str, destination: FilePath = None, progress_bar=True):
    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    try:
        if progress_bar:
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=destination) as t:
                filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
        else:
            filename, _ = urlretrieve(url, filename=destination)
    except (HTTPError, URLError, ValueError) as e:
        raise e
    return filename
