from chemdes.schema import GeneralReaction, ReactionCondition, ReactantSolution, ReactantSolvent, group_reactions, \
    Molecule


class ReactionNcOneLigand(GeneralReaction):
    def __init__(
            self,
            identifier: str,
            nano_crystal: ReactantSolution,
            ligand: ReactantSolution,
            conditions: [ReactionCondition],
            solvent: ReactantSolvent,
            properties: dict = None
    ):
        super().__init__(identifier, [ligand, solvent, nano_crystal], conditions, properties)
        self.solvent = solvent
        self.nano_crystal = nano_crystal
        self.ligand = ligand

    def is_reaction_NC_reference(self):
        """ whether the reaction is a reference reaction in which only NC solution and solvent were added """
        return self.nano_crystal.volume > 0 and self.ligand.volume < 1e-7 and self.solvent.volume > 0

    def is_reaction_blank_reference(self):
        """ whether the reaction is a reference reaction in which only solvent was added """
        return self.nano_crystal.volume < 1e-7 and self.ligand.volume < 1e-7 and self.solvent.volume > 0


def reactions_to_xy(reactions: [ReactionNcOneLigand]):
    x = []
    y = []
    x_units = []
    for r in reactions:
        r: ReactionNcOneLigand
        x.append(r.ligand.concentration * r.ligand.volume)
        y.append(r.properties["fom"])
        x_unit = "{}*{}".format(r.ligand.concentration_unit, r.ligand.volume_unit)
        x_units.append(x_unit)
    assert len(set(x_units)) == 1
    return x, y, x_units[0]


def categorize_reactions(reactions: [ReactionNcOneLigand]) -> dict[Molecule, list[[ReactionNcOneLigand]]]:
    unique_ligands, reaction_groups = group_reactions(reactions, "ligand.identity")
    ligand_to_categorized_reactions = dict()
    for ligand, reaction_group in zip(unique_ligands, reaction_groups):
        real_reactions = []
        ref_reactions = []
        blank_reactions = []
        for r in reaction_group:
            r: ReactionNcOneLigand
            if r.is_reaction_NC_reference():
                ref_reactions.append(r)
            elif r.is_reaction_blank_reference():
                blank_reactions.append(r)
            else:
                real_reactions.append(r)
        ligand_to_categorized_reactions[ligand] = [real_reactions, ref_reactions, blank_reactions]
    return ligand_to_categorized_reactions
