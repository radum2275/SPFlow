

import logging
from spn.structure.Base import Leaf, Product, Sum, InterfaceSwitch, assign_ids

from spn.algorithms.SPMN import SPMN, SPMNParams

from spn.algorithms.RSPMNHelper import get_partial_order_two_time_steps, get_feature_names_two_time_steps \
    , get_nodes_two_time_steps

from spn.algorithms.RSPMNInitialTemplateBuild import RSPMNInitialTemplate

from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import get_nodes_by_type
import numpy as np

from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.RSPMNUtil import eval_template_top_down


class RSPMN:

    def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names,
                 cluster_by_curr_information_set=False, util_to_bin=False):

        self.params = RSPMNParams(partial_order, decision_nodes, utility_nodes, feature_names,
                                  cluster_by_curr_information_set, util_to_bin)
        self.InitialTemplate = RSPMNInitialTemplate(self.params)
        self.template = None



class RSPMNParams:

    def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names, cluster_by_curr_information_set, util_to_bin):
        self.partial_order = partial_order
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.feature_names = feature_names
        self.cluster_by_curr_information_set = cluster_by_curr_information_set
        self.length_of_time_slice = len(self.feature_names)
        self.util_to_bin = util_to_bin