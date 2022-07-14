import heapq
import itertools
import math
import random
from typing import Tuple

import numpy as np

from lsal.schema import Molecule


def ks_sampler(dmat: np.ndarray, k: int) -> [int]:
    """
    taken from https://github.com/karoka/Kennard-Stone-Algorithm

    :param dmat: distance matrix, dmat[i][j] gives the distance between ith sample and jth sample
    :param k: number of samples we want to select from n (population size)
    :return: a list of sampled indices
    """
    assert dmat.ndim == 2
    assert dmat.shape[0] == dmat.shape[1]
    n = dmat.shape[0]
    assert k <= n
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


def indices_to_sample_list(population, indices: list[int]):
    return [population[i] for i in indices]


def pair_indices_to_sample_list(population, pair_indices: list[Tuple[int, ...]]):
    return [(population[i], population[j]) for i, j in pair_indices]


class MoleculeSampler:

    def __init__(self, population: list[Molecule], dmat: np.ndarray):
        self.population = population
        assert len(self.population) > 1, "the population should have a size at least 2, we got: {}".format(
            len(self.population))
        self.indices = list(range(len(self.population)))
        self.pair_indices = list(itertools.combinations(self.indices, 2))
        assert len(self.pair_indices) == len(set(self.pair_indices)) == math.comb(len(self.indices), 2)
        self.dmat = dmat

    def sample_random(self, k: int = None, seed: int = 42, return_mol=True):
        """
        randomly select k **unique** samples from the population
        """
        if k is None:
            k = len(self.population)
        rs = random.Random(x=seed)
        sampled_indices = rs.sample(self.indices, k=k)
        if return_mol:
            return indices_to_sample_list(self.population, sampled_indices)
        else:
            return sampled_indices

    def psample_random(self, k: int = None, seed: int = 42, return_mol=True):
        """
        randomly select k **unique** pairs from the population
        """
        if k is None:
            k = len(self.population)
        rs = random.Random(x=seed)
        sampled_indices = rs.sample(self.pair_indices, k=k)
        if return_mol:
            return pair_indices_to_sample_list(self.population, sampled_indices)
        else:
            return sampled_indices

    def sample_ks(self, k: int = None, return_mol=True):
        """
        randomly select k **unique** samples from the population
        """
        if k is None:
            k = len(self.population)

        sampled_indices = ks_sampler(self.dmat, k)
        if return_mol:
            return indices_to_sample_list(self.population, sampled_indices)
        else:
            return sampled_indices

    def psample_ks(self, k: int = None, return_mol=True, pdist="sum_of_two_smallest"):
        if k is None:
            k = len(self.pair_indices)
        pdmat, pid_to_pair = dmat_mol_to_dmat_pair(self.dmat, pdist)
        sampled_pindices = [pid_to_pair[pid] for pid in ks_sampler(pdmat, k)]
        if return_mol:
            return pair_indices_to_sample_list(self.population, sampled_pindices)
        else:
            return sampled_pindices

    def psample_ipd(self, k: int = None, return_mol=True):  # in-pair distance
        sorted_pindices = sorted(self.pair_indices, key=lambda x: self.dmat[x[0]][x[1]], reverse=True)
        sampled_pindices = sorted_pindices[:k]
        if return_mol:
            return pair_indices_to_sample_list(self.population, sampled_pindices)
        else:
            return sampled_pindices
