import collections
import copy
import logging

from spn.algorithms.EM import get_node_updates_for_EM
# from spn.algorithms.Gradient import gradient_backward
from spn.structure.Base import Leaf, Sum, InterfaceSwitch, assign_ids

from spn.algorithms.SPMN import SPMN, SPMNParams

from spn.algorithms.RSPMN.RSPMNInitialTemplateBuild import RSPMNInitialTemplate

from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import get_nodes_by_type
import numpy as np

from spn.algorithms.MPE import get_node_funtions, mpe
from spn.algorithms.RSPMN.TemplateUtil import eval_template_top_down, \
    gradient_backward

from spn.algorithms.MEU import meu_max
from spn.structure.Base import Max

from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface

from spn.algorithms.Gradient import get_node_gradients

from spn.algorithms.MEU import meu


class RSPMN:

    def __init__(self, partial_order, decision_nodes, utility_nodes,
                 feature_names, meta_types,
                 cluster_by_curr_information_set=False,
                 util_to_bin=False):

        self.params = RSPMNParams(partial_order, decision_nodes, utility_nodes,
                                  feature_names, meta_types,
                                  cluster_by_curr_information_set,
                                  util_to_bin
                                  )
        self.InitialTemplate = RSPMNInitialTemplate(self.params)
        self.template = None

    # Import methods of RSPMN class
    from ._RSMPNHardEM import eval_rspmn_bottom_up, eval_rspmn_top_down, learn_rspmn
    from ._RSPMNSoftEM import rspmn_gradient_backward, EM_optimization
    from ._RSPMNMeu import meu, eval_rspmn_bottom_up_for_meu

    # Other methods of class
    def get_params_for_get_each_time_step_data_for_template(self,
                                                            template, data):

        num_variables_each_time_step = len(self.params.feature_names)
        total_num_of_time_steps = int(
            data.shape[1] / num_variables_each_time_step)
        initial_num_latent_interface_nodes = len(template.children)

        return num_variables_each_time_step, \
               total_num_of_time_steps, initial_num_latent_interface_nodes

    @staticmethod
    def get_each_time_step_data_for_template(data, time_step_num,
                                             total_num_of_time_steps,
                                             eval_val_per_node,
                                             initial_num_latent_interface_nodes,
                                             num_variables_each_time_step,
                                             bottom_up=True):

        each_time_step_data = data[:,
                              (time_step_num * num_variables_each_time_step):
                              (
                                      time_step_num * num_variables_each_time_step) + num_variables_each_time_step]

        assert each_time_step_data.shape[1] == num_variables_each_time_step

        if time_step_num == total_num_of_time_steps - 1:
            # last time step in the sequence. last level of template
            # latent node data is corresponding bottom time step interface
            # node's inference value. 1 for last level
            latent_node_data = np.zeros((each_time_step_data.shape[0],
                                         initial_num_latent_interface_nodes))

        else:
            if bottom_up:
                # latent node data is corresponding bottom time step
                # interface node's inference value
                latent_node_data = eval_val_per_node[:,
                                   1:initial_num_latent_interface_nodes + 1]
            else:
                # latent node data top down is curr time step lls_
                # per_node last set of values
                first_latent_node_column = \
                    eval_val_per_node.shape[1] - \
                    initial_num_latent_interface_nodes
                latent_node_data = \
                    eval_val_per_node[:, first_latent_node_column:]

        each_time_step_data_for_template = np.concatenate(
            (each_time_step_data, latent_node_data), axis=1)

        return each_time_step_data_for_template

    @staticmethod
    def get_each_time_step_data_for_meu(data, time_step_num,
                                             total_num_of_time_steps,
                                             eval_val_per_node,
                                             initial_num_latent_interface_nodes,
                                             num_variables_each_time_step,
                                             bottom_up=True):

        each_time_step_data = data[:,
                              (time_step_num * num_variables_each_time_step):
                              (
                                      time_step_num * num_variables_each_time_step) + num_variables_each_time_step]

        assert each_time_step_data.shape[1] == num_variables_each_time_step

        if time_step_num == total_num_of_time_steps - 1:
            # last time step in the sequence. last level of template
            # latent node data is corresponding bottom time step interface
            # node's inference value. 1 for last level
            latent_node_data = np.zeros((each_time_step_data.shape[0],
                                         initial_num_latent_interface_nodes))

        else:
            if bottom_up:
                # latent node data is corresponding bottom time step
                # interface node's inference value
                latent_node_data = eval_val_per_node[:,
                                   1:initial_num_latent_interface_nodes + 1]
                latent_node_data = np.log(latent_node_data)
            else:
                # latent node data top down is curr time step lls_
                # per_node last set of values
                first_latent_node_column = \
                    eval_val_per_node.shape[1] - \
                    initial_num_latent_interface_nodes
                latent_node_data = \
                    eval_val_per_node[:, first_latent_node_column:]

        each_time_step_data_for_template = np.concatenate(
            (each_time_step_data, latent_node_data), axis=1)

        return each_time_step_data_for_template

    @staticmethod
    def pass_meu_val_to_latent_interface_leaf_nodes(
            eval_val_per_node, prev_eval_val_per_node,
            num_of_template_children,
            latent_node_list
    ):

        for i in range(0,len(latent_node_list),num_of_template_children):
            for j in range(i, i+num_of_template_children):
                k = j % num_of_template_children
                eval_val_per_node[:, latent_node_list[j].id] = \
                    prev_eval_val_per_node[:, k+1]

        # # leaf latent node columns in eval_val_per_node correspond to
        # # last few columns in current time step. They are equal to
        # # num of latent interface nodes
        # first_latent_node_column = \
        #     eval_val_per_node.shape[1] - \
        #     num_of_template_children
        #
        # # eval val of last columns corr to latent interface nodes of
        # # current time step are equal to first columns corr to
        # # eval vals of prev time step
        # eval_val_per_node[:, first_latent_node_column:] = \
        #     prev_eval_val_per_node[:, 1:num_of_template_children + 1]
        print(f'meu at pass meu {eval_val_per_node}')
        return eval_val_per_node

    @staticmethod
    def update_weights(template):

        nodes = get_nodes_by_type(template)

        for node in nodes:

            if isinstance(node, Sum):

                if all(isinstance(child, LatentInterface) for child in
                       node.children):

                    for i, child in enumerate(node.children):
                        node.weights[i] = (node.children[i].count / node.count)

                    node.weights = (np.array(node.weights) / np.sum(
                        node.weights)).tolist()

                    print(node.weights)

            print(f'node {node}, count {node.count}')

    def log_likelihood(self, template, data):

        unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(template,
                                                                  data, True)
        # ll at root node
        log_likelihood = unrolled_network_lls_per_node[-1][:, 0]

        return log_likelihood, unrolled_network_lls_per_node


class RSPMNParams:

    def __init__(self, partial_order, decision_nodes, utility_nodes,
                 feature_names, meta_types,
                 cluster_by_curr_information_set, util_to_bin):
        self.partial_order = partial_order
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.feature_names = feature_names
        self.meta_types = meta_types
        self.cluster_by_curr_information_set = cluster_by_curr_information_set
        self.length_of_time_slice = len(self.feature_names)
        self.util_to_bin = util_to_bin
