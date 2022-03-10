import heapq
import itertools
import math
import random

from chemdes import *


def write_mols(mols: [Molecule], fn: typing.Union[str, pathlib.Path]):
    records = []
    for m in mols:
        record = {"ligand0": m.iupac_name}
        records.append(record)
    df = pd.DataFrame.from_records(records)
    df.to_csv(fn, index=False)


def write_pairs(pairs, fn: typing.Union[str, pathlib.Path]):
    records = []
    for p in pairs:
        p = tuple(p)
        m0, m1 = p
        record = {"ligand0": m0.iupac_name, "ligand1": m1.iupac_name}
        records.append(record)
    df = pd.DataFrame.from_records(records)
    df.to_csv(fn, index=False)


def ks_sampler(dmat: np.ndarray, k: int) -> [int]:
    """https://github.com/karoka/Kennard-Stone-Algorithm"""
    assert dmat.ndim == 2
    assert len(set(dmat.shape)) == 1
    n = dmat.shape[0]
    i0, i1 = np.unravel_index(np.argmax(dmat, axis=None), dmat.shape)
    selected = [i0, i1]
    k -= 2
    # iterate find the rest
    minj = i0
    while k > 0 and len(selected) < n:
        mindist = 0.0
        for j in range(n):
            if j not in selected:
                mindistj = min([dmat[j][i] for i in selected])
                if mindistj > mindist:
                    minj = j
                    mindist = mindistj
        if minj not in selected:
            selected.append(minj)
        k -= 1
    return selected


def sum_of_four(a, b, c, d):
    return sum([a, b, c, d])


def sum_of_two_smallest(a, b, c, d):
    return sum(heapq.nsmallest(2, [a, b, c, d]))


def dmat_mol_to_dmat_pair(dmat_mol: np.ndarray, pair_dist_def="sum_of_four"):
    """define a distance for molecular pairs"""
    if pair_dist_def == "sum_of_four":
        calc_pair_dist = sum_of_four
    elif pair_dist_def == "sum_of_two_smallest":
        calc_pair_dist = sum_of_two_smallest
    else:
        raise NotImplementedError
    n_mols = dmat_mol.shape[0]
    pair_indices = [p for p in itertools.combinations(range(n_mols), 2)]
    n_pairs = len(pair_indices)
    dmat_pair = np.zeros((n_pairs, n_pairs))
    pair_idx_to_pair = dict(zip(range(n_pairs), pair_indices))
    for i in range(dmat_pair.shape[0]):
        pair_i_a, pair_i_b = pair_idx_to_pair[i]
        for j in range(i, dmat_pair.shape[1]):
            pair_j_a, pair_j_b = pair_idx_to_pair[j]
            d_ia_ja = dmat_mol[pair_i_a][pair_j_a]
            d_ia_jb = dmat_mol[pair_i_a][pair_j_b]
            d_ib_ja = dmat_mol[pair_i_b][pair_j_a]
            d_ib_jb = dmat_mol[pair_i_b][pair_j_b]
            # should all be non-negative
            # note under the `sum_of_four` definition (a, b) - (a, b) can be > 0
            dmat_pair[i][j] = calc_pair_dist(d_ia_ja, d_ia_jb, d_ib_ja, d_ib_jb)
            dmat_pair[j][i] = dmat_pair[i][j]
    return dmat_pair, pair_idx_to_pair


if __name__ == '__main__':

    SEED = 42
    NSAMPLES_PAIR = None
    NSAMPLES_MOL = None

    data = json_load("dimred/dimred.json")
    dmat = data["dmat"]
    data_2d = data["data_2d"]
    molecules = data["molecules"]
    all_pairs = [frozenset(pair) for pair in itertools.combinations(molecules, 2)]
    assert len(all_pairs) == len(set(all_pairs)) == math.comb(len(molecules), 2)

    write_mols(molecules, "sampler/all_mols.csv")
    write_pairs(all_pairs, "sampler/all_pairs.csv")

    if NSAMPLES_MOL is None:
        NSAMPLES_MOL = len(molecules)
    if NSAMPLES_PAIR is None:
        NSAMPLES_PAIR = len(all_pairs)

    # random sample 1 mol
    random.seed(SEED)
    random_mols = random.sample(molecules, k=NSAMPLES_MOL)
    write_mols(random_mols, "sampler/random_mols.csv")

    # random sample pairs
    random.seed(SEED + 1)
    random_pairs = random.sample(all_pairs, k=NSAMPLES_PAIR)
    write_pairs(random_pairs, "sampler/random_pairs.csv")

    # ks sample 1 mol
    ks_mols = ks_sampler(dmat, k=NSAMPLES_MOL)
    write_mols([molecules[i] for i in ks_mols], "sampler/ks_mols.csv")

    # ks sample pairs, `sum_of_four` xor `sum_of_two_smallest`
    for distance_definition in ("sum_of_four", "sum_of_two_smallest"):
        dmat_pair_sum_of_four, pair_idx_to_pair = dmat_mol_to_dmat_pair(dmat, distance_definition)
        ks_pairs = []
        for pid in ks_sampler(dmat_pair_sum_of_four, NSAMPLES_PAIR):
            i, j = pair_idx_to_pair[pid]
            ks_pairs.append(frozenset([molecules[i], molecules[j]]))
        assert len(ks_pairs) == len(set(ks_pairs))
        write_pairs(ks_pairs, "sampler/ks_pairs-{}.csv".format(distance_definition))
