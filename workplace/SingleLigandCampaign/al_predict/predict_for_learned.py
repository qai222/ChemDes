from lsal.campaign import SingleWorkflow
from lsal.utils import pkl_dump, get_basename
from lsal.utils import pkl_load

if __name__ == '__main__':
    predictions_for_learned = dict()
    for fomdef in ["fom1", "fom2", "fom3"]:
        swf = pkl_load("../al_workflow/models/onepot/{}--std--notune--all.pkl".format(fomdef))
        swf: SingleWorkflow
        predictions = swf.learner.predict(swf.learner.learned_ligands, swf.amounts, swf.ligand_to_des_record)
        predictions_for_learned[fomdef] = predictions
    pkl_dump(predictions_for_learned, get_basename(__file__) + ".pkl", print_timing=False)
