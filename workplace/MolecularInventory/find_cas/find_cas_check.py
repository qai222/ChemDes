"""
all ligands in the seed dataset should present in `inchi_to_cas`
"""
import pandas as pd
from loguru import logger

from lsal.utils import json_load, json_dump

if __name__ == '__main__':
    seed_dataset_identifiers = pd.read_csv("../initial_dataset/init_inv.csv")['identifier']
    i2c = json_load("inchi_to_cas.json.gz", gz=True)
    i2c.update(
        {
            'InChI=1S/C8H16O2/c1-2-3-4-5-6-7-8(9)10/h2-7H2,1H3,(H,9,10)': '124-07-2',
            'InChI=1S/C18H36O2/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18(19)20/h2-17H2,1H3,(H,19,20)': '57-11-4',
        }
    )

    for inchi in seed_dataset_identifiers:
        try:
            cas = i2c[inchi]
            assert cas is not None
        except (AssertionError, KeyError) as e:
            logger.info(f'cannot find cas for: {inchi}')
            continue
    json_dump(i2c, 'inchi_to_cas.json.gz', gz=True)
