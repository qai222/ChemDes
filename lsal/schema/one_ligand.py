from __future__ import annotations

from lsal.schema.material import Molecule
from lsal.schema.reaction import GeneralReaction, ReactionCondition, ReactantSolution, ReactantSolvent


class ReactionOneLigand(GeneralReaction):
    def __init__(
            self,
            identifier: str,
            conditions: [ReactionCondition],
            solvent: ReactantSolvent,
            nc_solution: ReactantSolution = None,
            ligand_solution: ReactantSolution = None,
            properties: dict = None
    ):
        super().__init__(identifier, [ligand_solution, solvent, nc_solution], conditions, properties)
        self.solvent = solvent
        self.nc_solution = nc_solution
        self.ligand_solution = ligand_solution

    @property
    def is_reaction_nc_reference(self) -> bool:
        """ whether the reaction is a reference reaction in which only NC solution and solvent were added """
        nc_good = self.nc_solution is not None and self.nc_solution.volume > 0
        no_ligand = self.ligand_solution is None or self.ligand_solution.volume < 1e-7
        solvent_good = self.solvent is not None and self.solvent.volume > 0
        return nc_good and solvent_good and no_ligand

    @property
    def is_reaction_blank_reference(self):
        """ whether the reaction is a reference reaction in which only solvent was added """
        no_nc = self.nc_solution is None or self.nc_solution.volume < 1e-7
        no_ligand = self.ligand_solution is None or self.ligand_solution.volume < 1e-7
        solvent_good = self.solvent is not None and self.solvent.volume > 0
        return no_nc and no_ligand and solvent_good

    @staticmethod
    def group_by_ligand(reactions: list[ReactionOneLigand]) -> dict[Molecule, list[ReactionOneLigand]]:
        unique_ligands, reaction_groups = GeneralReaction.group_reactions(reactions, "ligand_solution.solute")
        return dict(zip(unique_ligands, reaction_groups))
