
### 2022.03.11
- The following ligands were found insoluble in m-xylene:
  ```
  DL-phenylalanine
  1,10-diaminodecane
  1,10-decyldiphosphnic acid
  ```
  We concluded it is better to remove them from random sampling, 
  but still keep them in the inventory.

### 2022.03.14
- The "blank" experiments can have non-zero PL fraction values. This results in
  the following warnings:
  ```
  CRITICAL:root:this BLANK reaction has nonzero FOM???
  CRITICAL:root:ReactionNcOneLigand:
      identifier: LS001_L5_robotinput--F06
      nano_crystal: {'@module': 'chemdes.schema', '@class': 'ReactantSolution', '@version': None, 'identity': 'CsPbBr3', 'volume': 0, 'concentration': None, 'solvent_identity': 'm-xylene', 'properties': {'batch': 'MK003'}, 'volume_unit': 'ul', 'concentration_unit': None}
      ligand: {'@module': 'chemdes.schema', '@class': 'ReactantSolution', '@version': None, 'identity': {'@module': 'chemdes.schema', '@class': 'Molecule', '@version': None, 'inchi': 'InChI=1S/C18H38S/c1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19/h19H,2-18H2,1H3', 'iupac_name': 'octadecane-1-thiol'}, 'volume': 0, 'concentration': 0.05, 'solvent_identity': 'm-xylene', 'properties': {}, 'volume_unit': 'ul', 'concentration_unit': 'M'}
      conditions: [{'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Temperature (C):', 'value': 25.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Stir Rate (rpm):', 'value': 750.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Mixing time1 (s):', 'value': 600.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Mixing time2 (s):', 'value': 600.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Reaction time (s):', 'value': 60.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Preheat Temperature (C):', 'value': 80.0}]
      solvent: {'@module': 'chemdes.schema', '@class': 'ReactantSolvent', '@version': None, 'identity': 'm-xylene', 'volume': 500, 'volume_unit': 'ul', 'properties': {}}
      properties: {'vial': 'F06', 'fom': 0.0011414656}
  CRITICAL:root:this BLANK reaction has nonzero FOM???
  CRITICAL:root:ReactionNcOneLigand:
      identifier: LS001_L2_robotinput--F12
      nano_crystal: {'@module': 'chemdes.schema', '@class': 'ReactantSolution', '@version': None, 'identity': 'CsPbBr3', 'volume': 0, 'concentration': None, 'solvent_identity': 'm-xylene', 'properties': {'batch': 'MK003'}, 'volume_unit': 'ul', 'concentration_unit': None}
      ligand: {'@module': 'chemdes.schema', '@class': 'ReactantSolution', '@version': None, 'identity': {'@module': 'chemdes.schema', '@class': 'Molecule', '@version': None, 'inchi': 'InChI=1S/C12H24O2/c1-2-3-4-5-6-7-8-9-10-11-12(13)14/h2-11H2,1H3,(H,13,14)', 'iupac_name': 'dodecanoic acid'}, 'volume': 0, 'concentration': 0.05, 'solvent_identity': 'm-xylene', 'properties': {}, 'volume_unit': 'ul', 'concentration_unit': 'M'}
      conditions: [{'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Temperature (C):', 'value': 25.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Stir Rate (rpm):', 'value': 750.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Mixing time1 (s):', 'value': 600.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Mixing time2 (s):', 'value': 600.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Reaction time (s):', 'value': 60.0}, {'@module': 'chemdes.schema', '@class': 'ReactionCondition', '@version': None, 'name': 'Preheat Temperature (C):', 'value': 80.0}]
      solvent: {'@module': 'chemdes.schema', '@class': 'ReactantSolvent', '@version': None, 'identity': 'm-xylene', 'volume': 500, 'volume_unit': 'ul', 'properties': {}}
      properties: {'vial': 'F12', 'fom': 0.00013574968}
  ```

### 2022.03.17
- sample concentrations for unlabelled ligands
  - denser >= 43 should be fine, p value estimates
- model persistence: once the active learning starts 
  the following parameters/feature set cannot change
- feature scaling method: do we want to use theoretic limits, or limits from dataset?
  - can be ignored in DT based methods
- error propagation
  - peak fitting?
- comparing arbitrary distributions:[pick top 10%, use KS/THOMPSON, fig 11. jcp2022]
  - A is higher than B
  - A is more uncertainly than B