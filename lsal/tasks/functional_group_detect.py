import os.path
import subprocess

from rdkit.Chem import MolFromSmiles, MolToMolFile
from tqdm import tqdm

from lsal.utils import FilePath, createdir, removefile, removefolder, file_exists

"""
use checkmol to detect functional groups given smiles

download and cite:
https://homepage.univie.ac.at/norbert.haider/cheminf/fgtable.pdf
https://homepage.univie.ac.at/norbert.haider/cheminf/cmmm.html

checkmol.exe should be in `lsal/bin/`
"""
_checkmol_binary = os.path.dirname(os.path.abspath(__file__))
_checkmol_binary = os.path.join(_checkmol_binary, "../bin/checkmol.exe")
assert file_exists(_checkmol_binary), f"not such file: {_checkmol_binary}"


def dfg(input_smis: list[str], tmp_folder: FilePath = "dfg_tmp", rmtmp=True):
    createdir(tmp_folder)
    dfg_data = dict()
    for i, smi in tqdm(enumerate(input_smis)):
        m = MolFromSmiles(smi)
        fn = "{0:06d}.mol".format(i)
        fn = os.path.join(tmp_folder, fn)
        MolToMolFile(m, fn)
        cmd = [_checkmol_binary, "-e", "-c", fn]
        result = subprocess.run(cmd, capture_output=True)
        data = result.stdout.decode("utf-8").strip()
        data = [fg.strip() for fg in data.split("\n")]

        if len(data) == 1 and data[0] == "":
            fg_literal = []
            fg_code = []
        else:
            fg_literal = data[:-1]
            fg_code = data[-1].split(";")[:-1]
        dfg_data[smi] = {"literal": fg_literal, "code": fg_code}
        if rmtmp:
            removefile(fn)
    if rmtmp:
        removefolder(tmp_folder)
    return dfg_data
