import logging

from lsal.tasks.sampler import MoleculeSampler, Molecule
from lsal.utils import json_load, FilePath


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


if __name__ == '__main__':
    # load dimred result
    dimensionality_reduction_result = json_load("../Inventory/dimred.json")
    distance_matrix = dimensionality_reduction_result["dmat"]  # this is the distmat after dimred
    ligand_molecules = dimensionality_reduction_result["molecules"]
    logging.warning("# of molecules loaded: {}".format(len(ligand_molecules)))

    # remove insoluble molecules
    insoluble_inchis = [
        "InChI=1S/C18H39O3P/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-22(19,20)21/h2-18H2,1H3,(H2,19,20,21)",
        "InChI=1S/C9H11NO2/c10-8(9(11)12)6-7-4-2-1-3-5-7/h1-5,8H,6,10H2,(H,11,12)",
        "InChI=1S/C10H24O6P2/c11-17(12,13)9-7-5-3-1-2-4-6-8-10-18(14,15)16/h1-10H2,(H2,11,12,13)(H2,14,15,16)"
    ]
    ligand_indices_to_include = []
    for i, m in enumerate(ligand_molecules):
        if m.inchi not in insoluble_inchis:
            ligand_indices_to_include.append(i)
    distance_matrix = distance_matrix[ligand_indices_to_include, :][:, ligand_indices_to_include]
    ligand_molecules = [ligand_molecules[i] for i in ligand_indices_to_include]
    logging.warning("# of molecules to sample: {}".format(len(ligand_molecules)))
    assert distance_matrix.shape[0] == len(ligand_molecules)
    write_molecules(ligand_molecules, "population.txt")

    # init sampler
    sampler = MoleculeSampler(ligand_molecules, distance_matrix)
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
