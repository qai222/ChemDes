import pandas as pd
from loguru import logger

from lsal.schema import Molecule, load_molecules, Worker
from lsal.tasks import calculate_cxcalc, calculate_mordred, dfg
from lsal.utils import get_basename, get_workplace_data_folder, get_folder, plot_molcloud, log_time, json_dump

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)


class InitialDataset(Worker):

    def __init__(
            self,

            # load_init_inventory
            init_ligand_sheet=f"2022_0217_ligand_InChI_mk.xlsx",
            init_solvent_sheet=f"2022_0217_solvent_InChI.csv",
            formatted_ligand_csv="init_inv.csv",
            formatted_solvent_csv="init_solvent_inv.csv",

            # calculate_descriptors
            ligand_descriptor_csv="init_des.csv",

            # molmap
            molcloud_figure="molmap.eps",

            # detect_functional_groups
            functional_groups_csv="ligand_functional_groups.csv"
    ):
        self.functional_groups_csv = functional_groups_csv
        self.molcloud_figure = molcloud_figure
        self.ligand_descriptor_csv = ligand_descriptor_csv
        self.formatted_solvent_csv = formatted_solvent_csv
        self.formatted_ligand_csv = formatted_ligand_csv
        self.init_solvent_sheet = init_solvent_sheet
        self.init_ligand_sheet = init_ligand_sheet
        super().__init__(name=self.__class__.__name__, code_dir=_code_folder, work_dir=_work_folder)

    @log_time
    def load_inventory(self):
        col_to_mol_kw = {'Name': 'name', 'IUPAC Name': 'iupac_name', 'InChI': 'identifier'}
        ligands = load_molecules(
            fn=self.init_ligand_sheet,
            col_to_mol_kw=col_to_mol_kw,
            mol_type='LIGAND'
        )
        solvents = load_molecules(
            fn=self.init_solvent_sheet,
            col_to_mol_kw=col_to_mol_kw,
            mol_type='SOLVENT'
        )
        json_dump(solvents, 'init_solvent_inv.json')
        json_dump(ligands, 'init_inv.json')
        Molecule.write_molecules(ligands, self.formatted_ligand_csv, output="csv")
        Molecule.write_molecules(solvents, self.formatted_solvent_csv, output="csv")

    @log_time
    def calculate_descriptors(self):
        mols = load_molecules(
            fn=self.formatted_ligand_csv,
            col_to_mol_kw={
                'label': 'label',
                'identifier': 'identifier',
                'smiles': 'smiles',
            },
            mol_type='LIGAND',
        )
        mordred_df = calculate_mordred(smis=[m.smiles for m in mols])
        cxcalc_smis, cxcalc_df = calculate_cxcalc(smis=[m.smiles for m in mols])
        assert len(cxcalc_smis) == len(mols)

        # # include pka if needed
        # pka_df = opera_pka("ligand_descriptors_OPERA2.7Pred.csv")
        # des_df = pd.concat([pka_df, cxcalc_df, mordred_df], axis=1)

        des_df = pd.concat([cxcalc_df, mordred_df], axis=1)

        descriptors = des_df.columns.tolist()
        logger.info(f"# of descriptors: {len(descriptors)}")
        descriptors = '\n'.join(descriptors)
        logger.info(f"list of descriptors: \n{descriptors}")
        des_df.to_csv(self.ligand_descriptor_csv, index=False)

    @log_time
    def molmap(self):
        inv_smis = pd.read_csv(self.formatted_ligand_csv)["smiles"].tolist()
        plot_molcloud(smis=inv_smis, width=15, outfile=self.molcloud_figure)

    @log_time
    def detect_functional_groups(self):
        ligand_df = pd.read_csv(self.formatted_ligand_csv)
        labels = ligand_df['label'].tolist()
        inv_smis = ligand_df['smiles'].tolist()
        data = dfg(inv_smis)
        records = []
        for k, v in data.items():
            r = {'smi': k}
            r.update(v)
            records.append(r)
        df = pd.DataFrame.from_records(records)
        df['label'] = labels
        df.to_csv(self.functional_groups_csv, index=False)


if __name__ == "__main__":
    worker = InitialDataset()
    worker.run(
        [
            'load_inventory',
            'calculate_descriptors',
            'molmap',
            'detect_functional_groups',
        ]
    )
