from loguru import logger

from lsal.schema import Worker, load_featurized_molecules, load_molecules
from lsal.tasks import FomCalculator, collect_reactions_l1
from lsal.utils import get_basename, get_folder, log_time, json_dump

_code_folder = get_folder(__file__)
_basename = get_basename(__file__)


class CollectReactions(Worker):

    def __init__(self, ):
        super().__init__(name=self.__class__.__name__, code_dir=_code_folder, work_dir="./")

        self.ligands = None
        self.solvents = None

    @log_time
    def load_molecules(self):
        ligands = load_featurized_molecules(
            "init_inv.csv", "init_des.csv", "LIGAND",
            col_to_mol_kw={
                'label': 'label',
                'identifier': 'identifier',
                'smiles': 'smiles',
            },
        )
        solvents = load_molecules(
            fn="init_solvent_inv.csv",
            col_to_mol_kw={
                'label': 'label',
                'identifier': 'identifier',
                'smiles': 'smiles',
                'name': 'name',
            },
            mol_type='SOLVENT'
        )
        self.ligands = ligands
        self.solvents = solvents

    @log_time
    def load_reactions(self):
        reaction_collection = collect_reactions_l1("sheets", self.ligands, self.solvents)
        FomCalculator(reaction_collection).update_foms()
        json_dump(reaction_collection, 'reaction_collection.json', gz=False)
        logger.info(reaction_collection.__repr__())


if __name__ == "__main__":
    worker = CollectReactions()
    worker.run(
        [
            'load_molecules',
            'load_reactions',
        ]
    )
