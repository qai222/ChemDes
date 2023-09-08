Files in this folder are for building and featurizing the molecular pool.
Molecules will be collected as `lsal.schema.material.Molecule` objects and serialized as `json`.
1. [initial_dataset/initial_dataset.py](initial_dataset/initial_dataset.py) for collecting ligands in the seed dataset;
2. [pubchem_screen/pubchem_screener.py](pubchem_screen/pubchem_screener.py) for screening PUBCHEM;
3. [find_cas/find_cas.py](find_cas/find_cas.py) for making a `inchi` to `cas number` mapping;
4. finally, run `ligands.py` to collect the pool as `ligands.json.gz`
