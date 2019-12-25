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

import spn.algorithms.MEU as spmnMeu


def meu(self, template, data):

    unrolled_network_meu_per_node = self.eval_rspmn_bottom_up_for_meu(template,
                                                              data, False)
    # ll at root node
    meu = unrolled_network_meu_per_node[-1][:, 0]

    return meu, unrolled_network_meu_per_node


def eval_rspmn_bottom_up_for_meu(self, template, data, *args):
    # assert self.InitialTemplate.top_network is not None,
    # f'top layer does not exist'
    # assert self.template is not None, f'template layer does not exist'

    assert type(data) is np.ndarray, 'data should be of type numpy array'

    num_variables_each_time_step, total_num_of_time_steps, \
        initial_num_latent_interface_nodes = \
        self.get_params_for_get_each_time_step_data_for_template(template,
                                                                 data)

    logging.debug(
        f'intial_num_latent_interface_nodes '
        f'{initial_num_latent_interface_nodes}')
    logging.debug(f'total_num_of_time_steps {total_num_of_time_steps}')

    template_nodes = get_nodes_by_type(template)
    latent_interface_list = []
    for node in template_nodes:
        if type(node) == LatentInterface:
            latent_interface_list.append(node)


    # for bottom most time step + 1
    likelihood_per_node = np.zeros((data.shape[0], len(template_nodes)))
    unrolled_network_likelihood_per_node = [likelihood_per_node]

    meu_per_node = np.zeros((data.shape[0], len(template_nodes)))
    unrolled_network_meu_per_node = [meu_per_node]

    # evaluate template bottom up at each time step
    for time_step_num_in_reverse_order in range(total_num_of_time_steps - 1,
                                                -1, -1):

        logging.debug(
            f'time_step_num_in_reverse_order '
            f'{time_step_num_in_reverse_order}')

        prev_likelihood_per_node = unrolled_network_likelihood_per_node[-1]
        logging.debug(f'prev_likelihood_per_node {prev_likelihood_per_node}')

        prev_meu_per_node = unrolled_network_meu_per_node[-1]
        logging.debug(f'prev_meu_per_node {prev_meu_per_node}')

        each_time_step_data_for_template = \
            self.get_each_time_step_data_for_meu(
                data,
                time_step_num_in_reverse_order,
                total_num_of_time_steps,
                prev_likelihood_per_node,
                initial_num_latent_interface_nodes,
                num_variables_each_time_step,
                bottom_up=True
            )

        if time_step_num_in_reverse_order == 0:

            top_nodes = get_nodes_by_type(self.InitialTemplate.top_network)
            meu_per_node = np.zeros((data.shape[0], len(top_nodes)))
            likelihood_per_node = np.zeros((data.shape[0], len(top_nodes)))

            top_latent_interface_list = []
            for node in top_nodes:
                if type(node) == LatentInterface:
                    top_latent_interface_list.append(node)

            self.pass_meu_val_to_latent_interface_leaf_nodes(
                meu_per_node, prev_meu_per_node,
                initial_num_latent_interface_nodes, top_latent_interface_list)

            spmnMeu.meu(self.InitialTemplate.top_network,
                                     each_time_step_data_for_template,
                                     meu_matrix=meu_per_node,
                                     lls_matrix=likelihood_per_node
                                     )

            # eval_val_per_node = meu_matrix
            print(f'meu_per_node {meu_per_node}')
            print(f'likelihood_per_node {likelihood_per_node}')
            # print(f'meu_matrix {meu_matrix}')

        else:
            meu_per_node = np.zeros((data.shape[0], len(template_nodes)))
            likelihood_per_node = np.zeros((data.shape[0], len(template_nodes)))

            self.pass_meu_val_to_latent_interface_leaf_nodes(
                meu_per_node, prev_meu_per_node,
                initial_num_latent_interface_nodes, latent_interface_list)

            spmnMeu.meu(template,
                                     each_time_step_data_for_template,
                                     meu_matrix=meu_per_node,
                                     lls_matrix=likelihood_per_node
                                     )

            # meu_per_node = meu_matrix
            print(f'meu_per_node {meu_per_node}')
            print(f'likelihood_per_node {likelihood_per_node}')

        unrolled_network_likelihood_per_node.append(likelihood_per_node)
        unrolled_network_meu_per_node.append(meu_per_node)

    print(unrolled_network_meu_per_node[-1][:, 0])

    return unrolled_network_meu_per_node


def best_next_decision(self, template, input_data):
    pass