import glob
from collections import OrderedDict

from lsal.campaign import SingleWorkflow
from lsal.utils import pkl_dump
from lsal.utils import pkl_load

if __name__ == '__main__':
    for swf_pkl in glob.glob("../models/obo/fom*obo.pkl"):
        swf = pkl_load(swf_pkl)
        swf: SingleWorkflow
        vis_data = OrderedDict()
        for key, status_data in swf.learner_history.items():
            vis_data[key] = swf.visualize_history_data(status_data)
        pkl_dump(vis_data, "visdata_obo/{}_visdata.pkl".format(swf.swf_name))
