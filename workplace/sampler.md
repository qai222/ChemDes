#### Molecules and pairs
- [all_mols.csv](sampler/all_mols.csv) is a list of all molecules (IUPAC names).
- [all_pairs.csv](sampler/all_pairs.csv) is a list of all molecular pairs.

#### Sampling
By default `smaple size = population size`.
- [random_mols.csv](sampler/random_mols.csv) random sampling result of molecules.
- [ks_mols.csv](sampler/ks_mols.csv) 
  kennard stone sampling result of molecules 
  based on manhattan distance matrix from expert selected features.
- [random_pairs.csv](sampler/random_pairs.csv) random sampling result of molecular pairs.
- [ks_pairs-sum_of_four.csv](sampler/ks_pairs-sum_of_four.csv)
  kennard stone sampling result of molecular pairs. The distance between the pair of (a, b) and the pair of (c, d)
  is the sum of D(a,c), D(a,d), D(b,c), D(b,d).
- [ks_pairs-sum_of_two_smallest.csv](sampler/ks_pairs-sum_of_two_smallest.csv)
  kennard stone sampling result of molecular pairs. The distance between the pair of (a, b) and the pair of (c, d)
  is the sum of the two smallest among D(a,c), D(a,d), D(b,c), D(b,d).