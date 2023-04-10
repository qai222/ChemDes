import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from lsal.schema import Worker, L1XReactionCollection
from lsal.tasks import CampaignLoader, BatchCheckerL1, BatchParams
from lsal.utils import get_basename, get_workplace_data_folder, get_folder, log_time, json_load, json_dump, \
    get_timestamp

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)


def fom_plot(rc: L1XReactionCollection, ncol=4, fillnan=True) -> plt.Figure:
    plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 18})

    lig_to_reactions = list(rc.ligand_to_reactions_mapping().items())
    nsubplots = len(lig_to_reactions)
    if nsubplots % ncol == 0:
        nrow = nsubplots // ncol
    else:
        nrow = 1 + nsubplots // ncol
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, figsize=(3 * ncol, 3 * nrow), sharex='all', sharey='all')

    for i in range(ncol * nrow):
        irow = i // ncol
        icol = i % ncol

        ax = axes[irow][icol]
        if i >= nsubplots:
            ax.set_visible(False)
            continue

        lig, reactions = lig_to_reactions[i]

        ref_reactions = [rr for rr in rc.ref_reactions if
                         rr.is_reaction_nc_reference and rr.batch_name == reactions[0].batch_name]
        ref_fom = np.mean([rr.properties['FigureOfMerit'] for rr in ref_reactions])
        ref_od = np.mean([rr.properties['OpticalDensity'] for rr in ref_reactions])

        amount_unit = [r.ligand_solution.amount_unit for r in reactions]
        assert len(set(amount_unit)) == 1
        amount_unit = amount_unit[0]

        sorted_reactions_by_amounts = sorted(reactions, key=lambda x: x.ligand_solution.amount)
        amounts = [r.ligand_solution.amount for r in sorted_reactions_by_amounts]
        foms = [r.properties['FigureOfMerit'] for r in sorted_reactions_by_amounts]
        ods = [r.properties['OpticalDensity'] for r in sorted_reactions_by_amounts]

        if fillnan:
            foms = np.nan_to_num(foms)
            ods = np.nan_to_num(ods)

        xlabel = f"Ligand amount ({amount_unit})"
        ylabel = f"Figure of merit"
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        ax.scatter(amounts, foms, c="b", label=lig.label, alpha=0.2)
        ax.axhline(ref_fom, xmin=0, xmax=1, ls="-", color="b")
        ax.set_title(lig.label)
        # ax.legend()
        ax_od = ax.twinx()
        ax_od.axhline(ref_od, xmin=0, xmax=1, ls="-", color="r")
        ax_od.scatter(amounts, ods, c="r", alpha=0.2, label=lig.label + "-OD")

        ax.set_xscale('log')

    fig.tight_layout()
    return fig


class ReactionCollector(Worker):

    def __init__(
            self,
            ligand_library_path=f"{_code_folder}/../../MolecularInventory/ligands.json.gz",
            solvent_library_path=f"{_code_folder}/../../MolecularInventory/initial_dataset/init_solvent_inv.json",
    ):
        super().__init__(name=self.__class__.__name__, code_dir=_code_folder, work_dir=_work_folder)

        self.solvent_library_path = solvent_library_path
        self.ligand_library_path = ligand_library_path

        self.solvent_library = json_load(self.solvent_library_path)
        self.ligand_library = json_load(ligand_library_path, gz=True)

        self.init_ligands = []
        self.pool_ligands = []
        for lig in self.ligand_library:
            if lig.mol_type == 'LIGAND':
                self.init_ligands.append(lig)
            elif lig.mol_type == 'POOL':
                self.pool_ligands.append(lig)
            else:
                raise ValueError(f"funny mol_type: {lig.mol_type}")

    @staticmethod
    def write_outputs(campaign_name: str, reaction_collection: L1XReactionCollection, check_data: dict):
        logger.warning(f"FINAL REACTION COLLECTION:\n{reaction_collection.__repr__()}")

        rc_json = f"{_work_folder}/reaction_collection_{campaign_name}.json.gz"
        json_dump(reaction_collection, rc_json, gz=True)

        check_json = f"{_work_folder}/reaction_collection_{campaign_name}_CHECK.json.gz"
        json_dump(check_data, check_json, gz=True)

        rc_png = f"{_work_folder}/reaction_collection_{campaign_name}.png"
        fig = fom_plot(reaction_collection, ncol=3)
        fig.savefig(rc_png, dpi=600)

        df = reaction_collection.as_dataframe()  # only real reactions are included
        rc_csv = f"{_work_folder}/reaction_collection_{campaign_name}.csv"
        df.to_csv(rc_csv, index=False)

        return [rc_json, rc_csv, rc_png, check_json]

    @log_time
    def load_reactions_sl0519(self):

        campaign_name = "SL0519"

        batch_params = BatchParams(
            ligand_identifier_convert={float(v): v for v in range(30)},
            ligand_identifier_type='int_label',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD390",
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.init_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            if "SL08" in bn:
                bc = BatchCheckerL1(
                    exclude_ligand_identifiers=["InChI=1S/C8H16O2/c1-2-3-4-5-6-7-8(9)10/h2-7H2,1H3,(H,9,10)", ]
                )
            else:
                bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )
        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )

    @log_time
    def load_reactions_al0907(self):

        campaign_name = "AL0907"

        al_0907_identities = """mf0005, Dioctyl ether, CCCCCCCCOCCCCCCCC
        mf0006, 4-(Dimethylamino)-1,2,2,6,6-pentamethylpiperidine, CN(C)C1CC(C)(C)N(C)C(C)(C)C1
        mf0007, Dihexyl ether, CCCCCCOCCCCCC
        mt0002, Dimethyl Tetrafluorosuccinate, COC(=O)C(F)(F)C(F)(F)C(=O)OC
        mt0007,	Diethyl dibromomalonate, CCOC(=O)C(Br)(Br)C(=O)OCC
        st0003,	Ethyl 4,4,5,5,5-Pentafluoro-3-oxovalerate, CCOC(=O)CC(=O)C(F)(F)C(F)(F)F
        mf0002,	4,4′-Trimethylenebis(1-methylpiperidine), CN1CCC(CCCC2CCN(C)CC2)CC1
        st0000,	5,6-Dihydro-1,4-dithiin-2,3-dicarboxylic Anhydride, O=C1OC(=O)C2=C1SCCS2
        mt0001, 2-Bromo-5-nitropyridine, O=[N+]([O-])c1ccc(Br)nc1
        mt0008, hexane-1-sulfonic acid, CCCCCCS(=O)(=O)O"""
        id_to_smiles = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in al_0907_identities.split("\n")}

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD390",
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )
        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )

    @log_time
    def load_reactions_al1026(self):

        campaign_name = "AL1026"

        al_1026_identities = """st0010,std-top2-true,2-Fluoro-3-(trifluoromethyl)benzaldehyde,14,,_01,,,,1024,112641-20-0
st0015,std-top2-true,2-Hydroxy-1-naphthaldehyde,19,,_01,,,,1024,708-06-5
st0016,std-top2-true,4′-(Trifluoromethyl)-2-biphenylcarboxylic acid,20,,_02,,,,1024,84392-17-6
st0017,std-top2-true,Diethyl 4-(trifluoromethyl)benzylphosphonate,21,,_02,,,,1024,99578-68-4
mt0016,mu-top2-true,5′-Bromo-2′-hydroxy-3′-nitroacetophenone,7,,_03,,,,1024,70978-54-0
mt0017,mu-top2-true,2-Nitro-4-(trifluoromethyl)benzaldehyde,8,,_03,,,,1024,109466-87-7
mt0010,mu-top2-true,Triethyl 1,3,5-benzenetricarboxylate,1,_01,,,,,1024,4105-92-4
st0007,std-top2-true,4-(Trifluoromethoxy)anisole,11,_01,,,,,1024,710-18-9
mt0011,mu-top2-true,N-Phenyl-bis(trifluoromethanesulfonimide),2,_02,,,,,1024,37595-74-7
st0011,std-top2-true,3-(Trifluoromethyl)benzhydrol,15,_02,,,,,1024,728-80-3
st0008,std-top2-true,Ethyl 2-(trifluoromethyl)thiazole-5carboxylate,12,_03,,,,,1024,131748-96-4
st0012,std-top2-true,3-[2-(Trifluoromethyl)phenyl]propionic acid,16,_03,,,,,1024,94022-99-8
st0009,std-top2-true,Bis(2,2,2-trifluoroethyl) methylphosphonate,13,_04,,,,,1024,757-95-9
st0013,std-top2-true,3-Bromobenzoic acid,17,_04,,,,,1024,585-76-2"""
        id_to_cas = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in al_1026_identities.split("\n")}

        cas_to_smiles = dict()
        for lig in self.ligand_library:
            for cas in lig.properties['cas_number']:
                cas_to_smiles[cas] = lig.smiles
        id_to_smiles = {k: cas_to_smiles[v] for k, v in id_to_cas.items()}

        # smiles_to_label = {lig.smiles: lig.label for lig in self.ligand_library}
        # id_to_label = {i: smiles_to_label[id_to_smiles[i]] for i in id_to_smiles}
        # import pprint
        # logger.critical(pprint.pformat(id_to_label))

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD",  # this is different from previous collectors
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )

        # exclude the outlier reaction in POOL-00042676
        outlier_reactions = [
            '2022_1024_AL1_04_st0009_st0013_robotinput@@G11',
        ]
        n_before_exclusion = len(reactions_campaign_passed)
        reactions_campaign_passed = [r for r in reactions_campaign_passed if r.identifier not in outlier_reactions]
        logger.critical(f"excluded reactions: {n_before_exclusion} --> {len(reactions_campaign_passed)}")

        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )

    @log_time
    def load_reactions_al1213(self):

        campaign_name = "AL1213"

        al_1213_identities = """st0021,std-true,Pyroquilon,3,001,,,,,1212,57369-32-1
st0023,std-top2-true,3-(Hydroxymethyl)-1-adamantol,6,001,,,,,1212,38584-37-1
st0025,std-top2-true,"2,6-Diphenylphenol",8,002,,,,,1212,2432-11-3
mt0020,mu-top2-true,"2,2'-Dihydroxy-4,4'-dimethoxybenzophenone",9,002,,,,,1212,131-54-4
mt0022,mu-top2-true,"2,2-Bis(4-hydroxy-3,5-dimethylphenyl)propane",11,003,,,,,1212,5613-46-7
mt0025,mu-top2-true,2-Bromobenzoic Acid,14,003,,,,,1212,88-65-3
mf0012,mu-top2-false,Dodecyl Methacrylate (stabilized with MEHQ),16,004,,,,,1212,142-90-5
mf0013,mu-top2-false,"2,3,4-(Trimethoxy)bromobenzene ",17,004,,,,,1212,10385-36-1
mf0011,mu-top2-false,"2,2,3,4,4,4-Hexafluorobutyl methacrylate",15,005,,,,,1212,36405-47-7"""
        id_to_cas = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in
                     al_1213_identities.split("\n")}

        cas_to_smiles = dict()
        for lig in self.ligand_library:
            for cas in lig.properties['cas_number']:
                cas_to_smiles[cas] = lig.smiles
        id_to_smiles = {k: cas_to_smiles[v] for k, v in id_to_cas.items()}

        # smiles_to_label = {lig.smiles: lig.label for lig in self.ligand_library}
        # id_to_label = {i: smiles_to_label[id_to_smiles[i]] for i in id_to_smiles}
        # import pprint
        # logger.critical(pprint.pformat(id_to_label))

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD",  # this is different from previous collectors
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )

        outlier_reactions = [
            # '2022_1024_AL1_04_st0009_st0013_robotinput@@G11',
        ]
        n_before_exclusion = len(reactions_campaign_passed)
        reactions_campaign_passed = [r for r in reactions_campaign_passed if r.identifier not in outlier_reactions]
        logger.critical(f"excluded reactions: {n_before_exclusion} --> {len(reactions_campaign_passed)}")

        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )

    @log_time
    def load_reactions_al1222(self):

        campaign_name = "AL1222"

        al_1222_identities = """03-001,845866-82-2
03-002,31252-42-3
03-009,819050-88-9
03-011,7163-25-9
03-015,1204295-87-3
03-029,84-72-0
03-004,4545-21-5
03-010,81477-94-3
03-027,828243-30-7
03-008,140-28-3"""
        id_to_cas = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in al_1222_identities.split("\n")}

        cas_to_smiles = dict()
        for lig in self.ligand_library:
            for cas in lig.properties['cas_number']:
                cas_to_smiles[cas] = lig.smiles
        id_to_smiles = {k: cas_to_smiles[v] for k, v in id_to_cas.items()}
        # smiles_to_label = {lig.smiles: lig.label for lig in self.ligand_library}
        # id_to_label = {i: smiles_to_label[id_to_smiles[i]] for i in id_to_smiles}
        # import pprint
        # logger.critical(pprint.pformat(id_to_label))

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD",  # this is different from previous collectors
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )

        outlier_reactions = [
            # '2022_1024_AL1_04_st0009_st0013_robotinput@@G11',
        ]
        n_before_exclusion = len(reactions_campaign_passed)
        reactions_campaign_passed = [r for r in reactions_campaign_passed if r.identifier not in outlier_reactions]
        logger.critical(f"excluded reactions: {n_before_exclusion} --> {len(reactions_campaign_passed)}")

        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )

    @log_time
    def load_reactions_al0130(self):

        campaign_name = "AL0130"

        al_0130_identities = """04-000,13680-35-8
04-003,104-66-5
04-008,606-28-0
04-009,1485-70-7
04-011,631-22-1
04-012,6789-88-4
04-013,2224-00-2"""
        id_to_cas = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in al_0130_identities.split("\n")}

        cas_to_smiles = dict()
        for lig in self.ligand_library:
            for cas in lig.properties['cas_number']:
                cas_to_smiles[cas] = lig.smiles
        id_to_smiles = {k: cas_to_smiles[v] for k, v in id_to_cas.items()}
        # smiles_to_label = {lig.smiles: lig.label for lig in self.ligand_library}
        # id_to_label = {i: smiles_to_label[id_to_smiles[i]] for i in id_to_smiles}
        # import pprint
        # logger.critical(pprint.pformat(id_to_label))

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD",  # this is different from previous collectors
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )

        outlier_reactions = [
            # '2022_1024_AL1_04_st0009_st0013_robotinput@@G11',
        ]
        n_before_exclusion = len(reactions_campaign_passed)
        reactions_campaign_passed = [r for r in reactions_campaign_passed if r.identifier not in outlier_reactions]
        logger.critical(f"excluded reactions: {n_before_exclusion} --> {len(reactions_campaign_passed)}")

        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )


    @log_time
    def load_reactions_al0303(self):

        campaign_name = "AL0303"

        al_0303_identities = """05-004,13680-35-8
05-010,98534-81-7
05-016,1143-38-0
05-012,327-75-3
05-017,91-40-7
05-013,15862-37-0
05-001,120-55-8
05-008,64415-14-1
05-009,14389-86-7
05-002,104-66-5
05-006,7305-59-1
05-011,64115-88-4"""
        id_to_cas = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in al_0303_identities.split("\n")}

        cas_to_smiles = dict()
        for lig in self.ligand_library:
            for cas in lig.properties['cas_number']:
                cas_to_smiles[cas] = lig.smiles
        id_to_smiles = {k: cas_to_smiles[v] for k, v in id_to_cas.items()}
        # smiles_to_label = {lig.smiles: lig.label for lig in self.ligand_library}
        # id_to_label = {i: smiles_to_label[id_to_smiles[i]] for i in id_to_smiles}
        # import pprint
        # logger.critical(pprint.pformat(id_to_label))

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD",  # this is different from previous collectors
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )

        outlier_reactions = [
            # '2022_1024_AL1_04_st0009_st0013_robotinput@@G11',
        ]
        n_before_exclusion = len(reactions_campaign_passed)
        reactions_campaign_passed = [r for r in reactions_campaign_passed if r.identifier not in outlier_reactions]
        logger.critical(f"excluded reactions: {n_before_exclusion} --> {len(reactions_campaign_passed)}")

        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )


    @log_time
    def load_reactions_al0331(self):

        campaign_name = "AL0331"

        al_0331_identities = """06-004,94651-33-9
06-018,4487-59-6
06-008,92-82-0
06-003,92-85-3
06-009,253-52-1
06-017,70-23-5
06-002,61390-40-7
06-001,391604-55-0
06-006,2627-35-2"""
        id_to_cas = {row.split(',')[0].strip(): row.split(',')[-1].strip() for row in al_0331_identities.split("\n")}

        cas_to_smiles = dict()
        for lig in self.ligand_library:
            for cas in lig.properties['cas_number']:
                cas_to_smiles[cas] = lig.smiles
        id_to_smiles = {k: cas_to_smiles[v] for k, v in id_to_cas.items()}
        # smiles_to_label = {lig.smiles: lig.label for lig in self.ligand_library}
        # id_to_label = {i: smiles_to_label[id_to_smiles[i]] for i in id_to_smiles}
        # import pprint
        # logger.critical(pprint.pformat(id_to_label))

        batch_params = BatchParams(
            ligand_identifier_convert=id_to_smiles,
            ligand_identifier_type='smiles',
            expt_input_columns=(
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)', 'Reagent5 (ul)',
                'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)', 'Reagent10 (ul)', 'Labware ID:',
                'Reaction Parameters', 'Parameter Values', 'Reagents', 'Reagent Name', 'Reagent Identity',
                'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns=(
                "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)",
            ),
            expt_input_condition_columns=(
                "Reaction Parameters", "Parameter Values",
            ),
            expt_input_vial_column="Vial Site",
            reagent_volume_unit='ul',
            reagent_concentration_unit='uM',
            reagent_possible_solvent=('m-xylene',),
            expt_output_vial_column="Unnamed: 0",
            expt_output_wall_tag_column_suffix="_wallTag",
            expt_output_od_column_suffix="_PL_OD",  # this is different from previous collectors
            expt_output_fom_column_suffix="_FOM"
        )

        campaign_loader = CampaignLoader(
            campaign_name=campaign_name,
            campaign_folder=f"{_work_folder}/{campaign_name}/",
            ligand_inventory=self.pool_ligands,
            solvent_inventory=self.solvent_library,
            batch_params=batch_params,
        )

        batch_checkers = dict()
        for bn in campaign_loader.batch_names:
            bc = BatchCheckerL1()
            batch_checkers[bn] = bc

        check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded = campaign_loader.load(
            batch_checkers=batch_checkers
        )

        outlier_reactions = [
            # '2022_1024_AL1_04_st0009_st0013_robotinput@@G11',
        ]
        n_before_exclusion = len(reactions_campaign_passed)
        reactions_campaign_passed = [r for r in reactions_campaign_passed if r.identifier not in outlier_reactions]
        logger.critical(f"excluded reactions: {n_before_exclusion} --> {len(reactions_campaign_passed)}")

        rc = L1XReactionCollection(reactions_campaign_passed)
        self.collect_files += self.write_outputs(
            campaign_name, rc, dict(
                check_msgs=check_msgs, reactions_campaign_discarded=reactions_campaign_discarded
            )
        )


if __name__ == "__main__":
    worker = ReactionCollector()
    for load_method in [
        # 'load_reactions_sl0519',
        # 'load_reactions_al0907',
        # 'load_reactions_al1026',
        # 'load_reactions_al1213',
        # 'load_reactions_al1222',
        # 'load_reactions_al0130',
        # 'load_reactions_al0303',
        'load_reactions_al0331',
    ]:
        log_file = f"{worker.code_dir}/{worker.name}-{load_method}-{get_timestamp()}.log"
        worker.run(
            [load_method, ], log_file=log_file
        )
        worker.final_collect()
        worker.collect_files = []
