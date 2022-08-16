import logging

import pandas as pd
from sklearn.metrics import pairwise_distances

from lsal.tasks.sampler import MoleculeSampler, Molecule
from lsal.utils import json_load, FilePath, scale_df, read_smi


def write_molecules(mols: list[Molecule], outfile: FilePath):
    with open(outfile, "w") as f:
        for m in mols:
            f.write("{} -- {}\n".format(m.label, m.iupac_name))


def write_molecule_pairs(mps: list, outfile: FilePath):
    with open(outfile, "w") as f:
        for i, (m1, m2) in enumerate(mps):
            f.write("pair: {0:03d}\n".format(i))
            f.write("{} -- {}\n".format(m1.label, m1.iupac_name))
            f.write("{} -- {}\n".format(m2.label, m2.iupac_name))
            f.write("\n")


def calculate_distance_matrix(descriptor_dataframe: pd.DataFrame, metric="manhattan", ):
    descriptor_dataframe = descriptor_dataframe.select_dtypes('number')
    df = scale_df(descriptor_dataframe)
    distance_matrix = pairwise_distances(df.values, metric=metric)
    return distance_matrix


def remove_inchi(ligands, descriptor_df, insoluble_inchis):
    ligand_indices_to_include = []
    for i, m in enumerate(ligands):
        if m.inchi not in insoluble_inchis:
            ligand_indices_to_include.append(i)
    descriptor_df = descriptor_df.iloc[ligand_indices_to_include]
    ligands = [ligands[i] for i in ligand_indices_to_include]
    return ligands, descriptor_df


if __name__ == '__main__':
    # load ligands and descriptors
    dataset_ligands = json_load("../MolecularInventory/seed_dataset/seed_dataset_ligand_list.json")
    dataset_descriptor_df = pd.read_csv(
        "../MolecularInventory/seed_dataset/seed_dataset_ligand_descriptor_2022_06_16.csv")
    assert len(dataset_ligands) == len(dataset_descriptor_df)

    # remove insoluble molecules
    insoluble_inchis = read_smi("insoluble_inchi.txt")
    soluble_ligands, soluble_descriptor_df = remove_inchi(dataset_ligands, dataset_descriptor_df,
                                                          insoluble_inchis)

    logging.warning("# of molecules loaded: {}".format(len(soluble_ligands)))
    distmat = calculate_distance_matrix(soluble_descriptor_df)
    logging.warning("# of molecules to sample: {}".format(len(soluble_ligands)))
    assert distmat.shape[0] == len(soluble_ligands)
    write_molecules(soluble_ligands, "population.txt")

    # init sampler
    sampler = MoleculeSampler(soluble_ligands, distmat)
    seed = 42
    # 1-ligand random sampling
    samples = sampler.sample_random(k=None, seed=seed, return_mol=True)
    write_molecules(samples, "1ligand_random.txt")
    # 1-ligand ks sampling
    samples = sampler.sample_ks(k=None, return_mol=True)
    write_molecules(samples, "1ligand_ks.txt")
    # 2-ligand random sampling
    samples = sampler.psample_random(k=None, seed=seed + 1, return_mol=True)
    write_molecule_pairs(samples, "2ligand_random.txt")
    # 2-ligand ks sampling
    samples = sampler.psample_ks(k=None, return_mol=True, pdist="sum_of_two_smallest")
    write_molecule_pairs(samples, "2ligand_ks.txt")
    # 2-ligand sort by in-pair distance
    samples = sampler.psample_ipd(k=None, return_mol=True)
    write_molecule_pairs(samples, "2ligand_ipd.txt")
