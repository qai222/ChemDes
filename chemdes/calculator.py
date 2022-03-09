import abc

import mordred
import pandas as pd
import rdkit.Chem.rdmolops
from monty.json import MSONable
from mordred import descriptors as mordred_descriptors

from chemdes.schema import Descriptor, Molecule

_mordred_descriptors = mordred.Calculator(mordred_descriptors).descriptors


class MoleculeCalculator(MSONable, abc.ABC):
    def __init__(self, name: str, targets: [Descriptor], results: dict = None):
        self.name = name
        self.targets = targets

        if results is None:
            results = dict()

        self.results = results

    @abc.abstractmethod
    def calc_one(self, mol: Molecule):
        pass

    @abc.abstractmethod
    def calc_all(self, mols: [Molecule]):
        pass


class MordredCalculator(MoleculeCalculator):
    def __init__(self, results: dict = None):
        """
        calculator using `mordred`, https://github.com/mordred-descriptor/mordred

        Moriwaki H, Tian Y-S, Kawashita N, Takagi T (2018)
        Mordred: a molecular descriptor calculator.
        Journal of Cheminformatics 10:4 .
        doi: 10.1186/s13321-018-0258-y

        :param results: dict[molecule][descriptor] -> value
        """
        super().__init__("MORDRED", targets=self.get_all_mordred_descriptors(), results=results)

    def calc_one(self, mol: Molecule) -> dict:
        self.results = dict()
        calc = mordred.Calculator(mordred_descriptors, ignore_3D=True)
        df = calc.pandas([mol.rdmol], quiet=True)  # missing values removed
        return self.mordred_df_to_results(df, [mol])

    def calc_all(self, mols: [Molecule]) -> None:
        self.results = dict()
        calc = mordred.Calculator(mordred_descriptors, ignore_3D=True)
        df = calc.pandas([m.rdmol for m in mols], quiet=False)
        self.results = self.mordred_df_to_results(df, mols)

    def get_target_by_name(self, name: str) -> Descriptor:
        for t in self.targets:
            if t.name == name:
                return t
        raise ValueError("descriptor not found!: {}".format(name))

    def mordred_df_to_results(self, df: pd.DataFrame, mols: [Molecule]) -> dict:
        results = dict()
        for m, record in zip(mols, df.to_dict(orient="records")):
            record = {self.get_target_by_name(k): v for k, v in record.items() if
                      isinstance(v, float) or isinstance(v, int)}
            results[m] = record
        return results

    @staticmethod
    def get_all_mordred_descriptors() -> [Descriptor]:
        return [Descriptor.from_mordred_descriptor(des) for des in _mordred_descriptors]


class PubChemCalculator(MoleculeCalculator):

    def __init__(self, targets: [Descriptor], results: dict = None):
        """
        retrieve data from pubchem

        :param targets: a list of descriptors the calculator is assigned with
        :param results: dict[molecule][descriptor] -> value
        """

        super().__init__("PUBCHEM", targets=targets, results=results)

    # TODO implement


class RdkitCalculator(MoleculeCalculator):
    _rdkit_descriptor_names = ["FormalCharge"]
    _rdkit_descriptors = [Descriptor(n, source="RDKIT") for n in _rdkit_descriptor_names]

    def __init__(self, targets: [Descriptor] = _rdkit_descriptors, results: dict = None):
        assert set([t.name for t in targets]).issubset(set(self._rdkit_descriptor_names))
        super().__init__("RDKIT", targets=targets, results=results)

    def calc_one(self, mol: Molecule):
        results = dict()
        results[mol] = dict()
        for target in self.targets:
            calc_function = getattr(self, "_calc_" + target.name)
            assert callable(calc_function)
            v = calc_function(mol)  # TODO add params
            results[mol][target] = v
        return results

    def calc_all(self, mols: [Molecule]):
        results = dict()
        for m in mols:
            r = self.calc_one(m)
            results.update(r)
        self.results = results

    @staticmethod
    def _calc_FormalCharge(mol: Molecule) -> int:
        return rdkit.Chem.rdmolops.GetFormalCharge(mol.rdmol)


class OperaCalculator(MoleculeCalculator): pass
# _opera_descriptor_names = [
#     'logBCF', 'BP', 'logP', 'MP',
#     'logVP', 'WS', 'AOH', 'BioDeg', 'ReadyBiodeg', 'logHL', 'logKM',
#     'KOA', 'logKoc', 'RT', 'pKa', 'logD', 'CERAPP', 'CoMPARA', 'AcuteTox', 'FuB',
#     'Clint'
# ]
#
# # _cmd_opera = "C:\Windows\System32\cmd.exe /k set "PATH=%PATH%;C:\Program Files\OPERA\application" & "C:\Program Files\OPERA\application\OPERA.exe""
#
# def __init__(self, targets: [Descriptor], results: dict = None):
#     assert set([t.name for t in targets]).issubset(set(self._opera_descriptor_names))
#     super().__init__("OPERA", targets=targets, results=results)
