import functools
import itertools
import json
import logging
import os
import re
import pathlib
import typing

import monty.json
import numpy as np
from rdkit.Chem import MolToSmiles, MolToInchi, MolFromSmiles, MolFromSmarts
from rdkit.Chem.inchi import MolFromInchi

SEED = 42

FilePath = typing.Union[pathlib.Path, os.PathLike, str]

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


def json_dump(o, fn: typing.Union[str, pathlib.Path]):
    with open(fn, "w") as f:
        json.dump(o, f, cls=monty.json.MontyEncoder)


def json_load(fn: typing.Union[str, pathlib.Path], warning=False):
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


def strip_extension(p: typing.Union[pathlib.Path, str]):
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

def get_basename(path: typing.Union[pathlib.Path, str]):
    return strip_extension(os.path.basename(os.path.abspath(path)))