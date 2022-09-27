import glob

import pandas as pd
from loguru import logger
from rdkit.Chem import MolFromSmiles, MolToSmiles, Descriptors
from tqdm import tqdm

from lsal.schema.workplace import Worker
from lsal.tasks.descriptor_calculator import cxcalc_parallel_calculate, calculate_mordred, \
    cxcalc_parallel_collect_results, calculate_cxcalc
from lsal.tasks.screen_molecule import delta_plot, domain_range, smi2poolinv
from lsal.utils import file_exists, has_isotope
from lsal.utils import get_workplace_data_folder
from lsal.utils import read_smi, get_basename, json_load
from lsal.utils import remove_stereo, parse_formula, write_smi, get_folder, createdir, log_time

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)


class PubChemScreener(Worker):

    def __init__(
            self,

            # collect_pubchem_compounds
            pubchem_compound_csv="PubChem_compound.csv",
            pubchem_vendors=('Sigma-Aldrich',),

            # smiles_screening
            smiles_screening_mwmax=400,  # max molecular weight
            allowed_elements=frozenset({"C", "H", "O", "N", "P", "S", "Br", "F", }),
            pubchem_compound_csv_after_smiles_screen="PubChem_compound_smiles_screened.csv",

            # descriptor_cxcalc
            cxcalc_nproc=1,
            cxcalc_chunksize=1000,
            cxcalc_wdir="cxcalc_wdir",
            cxcalc_input="des_cxcalc.smi",
            cxcalc_output="des_cxcalc.csv",

            # descriptor_mordred
            mordred_output="des_mordred.csv",

            # screen_by_des_range
            init_des_csv=f"{_work_folder}/../initial_dataset/init_des.csv",
            wdir_screen_by_des_range="screen_by_descriptor",

            # collect_results
            delta_string="0.40",
            des_csv='screened_des.csv',
            inv_csv='screened_inv.csv',
    ):

        super().__init__(name=self.__class__.__name__, code_dir=_code_folder, work_dir=_work_folder)
        self.inv_csv = inv_csv
        self.des_csv = des_csv
        self.delta_string = delta_string
        self.wdir_screen_by_des_range = wdir_screen_by_des_range
        self.init_des_csv = init_des_csv
        self.mordred_output = mordred_output
        self.cxcalc_nproc = cxcalc_nproc
        self.cxcalc_chunksize = cxcalc_chunksize
        self.cxcalc_output = cxcalc_output
        self.cxcalc_input = cxcalc_input
        self.cxcalc_wdir = cxcalc_wdir
        self.pubchem_compound_csv_after_smiles_screen = pubchem_compound_csv_after_smiles_screen
        self.allowed_elements = allowed_elements
        self.smiles_screening_mwmax = smiles_screening_mwmax
        self.pubchem_vendors = pubchem_vendors
        self.pubchem_compound_csv = pubchem_compound_csv

    @log_time
    def collect_pubchem_compounds(self):
        """
        two ways to download pubchem compounds related to a vendor/vendors
        - use direct search from ncbi, the url is:
            https://www.ncbi.nlm.nih.gov/pccompound?term=(%22has%20src%20vendor%22%5BFilter%5D)%20AND%20%22Sigma-Aldrich%22%5BSourceName%5D
        - use ChemScraper.download_vendor_compounds
        """
        if file_exists(self.pubchem_compound_csv):
            logger.warning(f"file already exists, download skipped: {self.pubchem_compound_csv}")
            return
        from ChemScraper import download_vendor_compounds
        download_vendor_compounds(
            vendors=self.pubchem_vendors,
            saveas=self.pubchem_compound_csv,
            # count_limit=70,  # comment out to download all entries
            field_string='cid,mw,mf,isosmiles',
        )

    @log_time
    def smiles_screening(self):
        df = pd.read_csv(self.pubchem_compound_csv)
        logger.info(f'loaded # of molecules: {len(df)}')
        keep_tuples = []
        for t in tqdm(df.itertuples(index=False, name=None)):
            cid, mw, formula, smiles = t
            # - more than one component
            if "." in smiles:
                continue
            # - apparent charge in formula
            if "+" in formula or "-" in formula:
                continue
            # - invalid formula
            try:
                fdict = parse_formula(formula)
            except ValueError:
                continue
            # - carbon should be there
            if "C" not in fdict.keys():
                continue
            # - not a subset of allowed elements
            if not set(fdict.keys()).issubset(self.allowed_elements):
                continue
            smiles = remove_stereo(smiles)
            # - invalid smiles
            try:
                m = MolFromSmiles(smiles)
            except ValueError:
                continue
            # - any isotope
            if has_isotope(m):
                continue
            # - mw larger than 400
            mw = Descriptors.ExactMolWt(m)
            if mw > self.smiles_screening_mwmax:
                continue
            smiles = MolToSmiles(m)
            keep_tuples.append((cid, smiles))
        df_screened = pd.DataFrame(keep_tuples, columns=['cid', 'smiles'])
        df_screened.drop_duplicates(subset=['smiles', ], inplace=True)
        df_screened.to_csv(self.pubchem_compound_csv_after_smiles_screen, index=False)
        logger.info(f'screened to # of molecules: {len(df_screened)}')
        return df_screened

    @log_time
    def descriptor_cxcalc(self):
        smis = pd.read_csv(self.pubchem_compound_csv_after_smiles_screen)['smiles']
        if len(glob.glob(f"{self.cxcalc_wdir}/*.out")) > 0 and len(glob.glob(f"{self.cxcalc_wdir}/*.smi")) > 0:
            logger.warning(f"use existing cxcalc output files in: {self.cxcalc_wdir}")
            in_files = sorted(glob.glob(f"{self.cxcalc_wdir}/*.smi"))
            out_files = sorted(glob.glob(f"{self.cxcalc_wdir}/*.out"))
            # combine_files(in_files, self.cxcalc_input)
            final_input_smis, df = cxcalc_parallel_collect_results(in_files, out_files)
        elif self.cxcalc_nproc == 1:
            if file_exists(self.cxcalc_input) and file_exists(self.cxcalc_output):
                logger.warning("using existing cxcalc serial output")
                return
            else:
                logger.warning(f"run cxcalc in serial mode")
                final_input_smis, df = calculate_cxcalc(smis=smis, remove_mol_file=False)
        else:
            createdir(self.cxcalc_wdir)
            logger.warning("run cxcalc in parallel")
            final_input_smis, df = cxcalc_parallel_calculate(
                smis=smis, workdir=self.cxcalc_wdir,
                combined_input=self.cxcalc_input, nproc=self.cxcalc_nproc, chunk_size=self.cxcalc_chunksize,
            )
        write_smi(final_input_smis, self.cxcalc_input)
        df.to_csv(self.cxcalc_output, index=False)

    @log_time
    def descriptor_mordred(self):
        if file_exists(self.mordred_output):
            logger.warning("using existing mordred output")
            return
        smis = read_smi(self.cxcalc_input)
        mordred_df = calculate_mordred(smis)
        mordred_df.to_csv(self.mordred_output, index=False)

    @log_time
    def screen_by_des_range(self):
        # load pool smis
        mordred_df = pd.read_csv(self.mordred_output)
        cxcalc_df = pd.read_csv(self.cxcalc_output)
        cxcalc_smis = read_smi(self.cxcalc_input)
        assert len(mordred_df) == len(cxcalc_df) == len(cxcalc_smis)
        pool_smis = cxcalc_smis

        # combine cxcalc and mordred
        cm_df = pd.concat([cxcalc_df, mordred_df], axis=1)
        available_features = cm_df.columns.tolist()
        logger.info("available features: {}".format(available_features))
        # smi2record = get_smi2record(pool_smis, cm_df, None)

        # delta plot
        createdir(self.wdir_screen_by_des_range)
        logger.info(f"screen against: {self.init_des_csv}")
        lim = domain_range(self.init_des_csv, available_features)
        delta_plot(
            pool_smis, lim, cm_df,
            f"{self.wdir_screen_by_des_range}/delta_des_plot.eps",
            available_features,
            wdir=self.wdir_screen_by_des_range,
            logger=logger,
        )

    @log_time
    def collect_results(self):
        records, smis = json_load(f"{self.wdir_screen_by_des_range}/{self.delta_string}.json")
        df_inv = smi2poolinv(smis)
        df_des = pd.DataFrame.from_records(records)
        df_des.to_csv(self.des_csv, index=False)
        df_inv.to_csv(self.inv_csv, index=False)


if __name__ == "__main__":
    worker = PubChemScreener()
    worker.run(
        [
            'collect_pubchem_compounds',
            'smiles_screening',
            'descriptor_cxcalc',
            'descriptor_mordred',
            'screen_by_des_range',
            'collect_results',
        ]
    )
