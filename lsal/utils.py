import collections
import functools
import itertools
import json
import logging
import os
import pathlib
import pickle
import re
import time
import typing
from datetime import datetime

import monty.json
import numpy as np
import pandas as pd
from monty.json import MSONable
from rdkit.Chem import MolToSmiles, MolToInchi, MolFromSmiles, MolFromSmarts
from rdkit.Chem.inchi import MolFromInchi
from sklearn import preprocessing

SEED = 42

FilePath = typing.Union[pathlib.Path, os.PathLike, str]


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


def json_dump(o, fn: FilePath):
    with open(fn, "w") as f:
        json.dump(o, f, cls=monty.json.MontyEncoder)


def json_load(fn: FilePath, warning=False):
    if warning:
        logging.warning("loading file: {}".format(fn))
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
        print("dumped {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))


def pkl_load(fn: FilePath, print_timing=True):
    ts1 = time.perf_counter()
    with open(fn, "rb") as f:
        d = pickle.load(f)
    ts2 = time.perf_counter()
    if print_timing:
        print("loaded {} in: {:.4f} s".format(os.path.basename(fn), ts2 - ts1))
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
    return abs(a - b) / min([aa, bb]) < eps


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
