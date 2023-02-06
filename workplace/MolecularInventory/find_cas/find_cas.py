import gzip
import re
from collections import defaultdict

from loguru import logger
from tqdm import tqdm

from lsal.schema import Worker
from lsal.utils import get_basename, get_workplace_data_folder, get_folder, log_time, json_dump, \
    FilePath, file_exists, download_file, get_file_size, json_load

"""
Find CAS RN using PubChem
"""

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)


class CasFinder(Worker):

    def __init__(
            self,
            inchis: list[str],
            pubchem_synonym_file: FilePath = f'{_work_folder}/CID-Synonym-filtered.gz',
            output: FilePath = 'cid_to_cas.json'
    ):
        super().__init__(name=self.__class__.__name__, code_dir=_code_folder, work_dir=_work_folder)
        self.inchis = inchis
        self.output = output
        self.pubchem_synonym_file = pubchem_synonym_file
        self.inchi_to_cid_json = f'{self.work_dir}/inchi_to_cid.json'
        self.cid_to_cas_json = f'{self.work_dir}/cid_to_cas.json'
        self.inchi_to_cas_json = f'{self.work_dir}/inchi_to_cas.json.gz'

    @log_time
    def download_synonymy_file(self):
        if not file_exists(self.pubchem_synonym_file):
            download_file(
                url='https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz',
                destination=self.pubchem_synonym_file,
                progress_bar=True
            )
        else:
            logger.warning(
                f'Found synonym file: {self.pubchem_synonym_file} of size {get_file_size(self.pubchem_synonym_file, unit="g")} GB')

    @log_time
    def find_cids(self):
        if file_exists(self.inchi_to_cid_json):
            logger.warning(f'Found inchi to cid json: {self.inchi_to_cid_json}')
            inchi_to_cid = json_load(self.inchi_to_cid_json, gz=False)
        else:
            import ChemScraper
            inchi_to_cid = ChemScraper.request_convert_identifiers(
                identifiers=self.inchis, input_type='inchi', output_type='cid'
            )
            json_dump(inchi_to_cid, self.inchi_to_cid_json, gz=False)
        logger.warning(
            f'no cid found from inchi for # of molecules: '
            f'{len([v for v in inchi_to_cid.values() if v is None])}'
            f' out of {len(self.inchis)}'
        )

    @log_time
    def cid_to_cas(self):
        """
        I tried two methods
        1. `ChemScraper.get_cas_number`, this send a request to grap `pug_view` of pubchem, then check the `leaf` nodes
            in the returned nested json, not suitable for large dataset (one cid per request)
            >> found/total: 11311/43724
        2. Pubchem synonym file contains all synonyms for a given cid, download it from NIH FTP server then use regex
            to grep CAS RN, this was developed in
            https://gist.github.com/KhepryQuixote/00946f2f7dd5f89324d8 by Khepry Quixote
            >> found/total: 29926/43724
        """
        if file_exists(self.cid_to_cas_json):
            logger.warning(f"Found cid_to_cas json: {self.cid_to_cas_json}")
        else:
            cas_pattern = b"^[1-9][0-9]{1,6}\\-[0-9]{2}\\-[0-9]$"
            # cas_regex = re.compile(pattern=)
            cid_to_cas = defaultdict(list)
            if self.pubchem_synonym_file.endswith('.gz'):
                open_file = gzip.open
            else:
                open_file = open
            with open_file(self.pubchem_synonym_file, 'rb') as f:
                for line in tqdm(f):
                    cid, synonym = line.split(b'\t')
                    if re.match(cas_pattern, synonym):
                        cid_to_cas[int(cid.decode("utf-8"))].append(synonym.rstrip(b'\n').decode("utf-8"))
            json_dump(cid_to_cas, self.cid_to_cas_json, gz=False)

    @log_time
    def inchi_to_cas(self):
        inchi_to_cas = dict()

        cid_to_cas = json_load(self.cid_to_cas_json, gz=False)
        inchi_to_cid = json_load(self.inchi_to_cid_json, gz=False)
        for inchi in tqdm(self.inchis):
            try:
                cas_list = cid_to_cas[inchi_to_cid[inchi]]
                inchi_to_cas[inchi] = cas_list
            except KeyError:
                inchi_to_cas[inchi] = None
        json_dump(inchi_to_cas, self.inchi_to_cas_json, gz=True)
        logger.warning(
            f"found cas/total: {len([v for v in inchi_to_cas.values() if v is not None])}/{len(inchi_to_cas)}")
        self.collect_files.append(self.inchi_to_cas_json)


if __name__ == "__main__":
    worker = CasFinder(
        inchis=[lig.identifier for lig in json_load("../ligands.json.gz", gz=True)],
    )
    worker.run(
        [
            'download_synonymy_file',
            'find_cids',
            'cid_to_cas',
            'inchi_to_cas',
        ]
    )
    worker.final_collect()
