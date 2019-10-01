

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

from spn.algorithms.MEU import meu_max
from spn.structure.Base import Max

from spn.structure.leaves.spmnLeaves.SPMNLeaf import LatentInterface


class RSPMN:

    def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names,
                 cluster_by_curr_information_set=False, util_to_bin=False):

        self.params = RSPMNParams(partial_order, decision_nodes, utility_nodes, feature_names,
                                  cluster_by_curr_information_set, util_to_bin)
        self.InitialTemplate = RSPMNInitialTemplate(self.params)
        self.template = None

    def learn_rspmn(self, data, total_num_of_time_steps_varies):

        if total_num_of_time_steps_varies:
            assert type(data) is list, 'When sequence length varies, data is a list of numpy arrays'
            print("Evaluating rspn and collecting nodes to update")

            for row in range(len(data)):
                each_data_point = data[row]
                # print("length of sequence:", self.get_len_sequence())

                unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(each_data_point)
        else:
            assert type(data) is np.ndarray, 'data should be of type numpy array'

            # print("Length of the sequence in mini_batch:", len_seq)
            # assert data.shape[
            #            1] == self.num_variables * self.len_sequence, "data columns not equal to number of variables time length of sequence"
            # # print("Evaluating rspn and collecting nodes to update")

            unrolled_network_lls_per_node = self.eval_rspmn_bottom_up(data)
            unrolled_network_lls_per_node = self.eval_rspmn_top_down(data, unrolled_network_lls_per_node)

    def eval_rspmn_bottom_up(self, data):

        # assert self.InitialTemplate.top_network is not None, f'top layer does not exist'
        # assert self.InitialTemplate.template_network is not None, f'template layer does not exist'

        assert type(data) is np.ndarray, 'data should be of type numpy array'

        num_variables_each_time_step = len(self.params.feature_names)
        total_num_of_time_steps = int(data.shape[1] / len(self.params.feature_names))
        initial_num_latent_interface_nodes = len(self.InitialTemplate.template_network.children)

        logging.debug(f'intial_num_latent_interface_nodes {initial_num_latent_interface_nodes}')
        logging.debug(f'total_num_of_time_steps {total_num_of_time_steps}')

        template_nodes = get_nodes_by_type(self.InitialTemplate.template_network)
        self.template = self.InitialTemplate.template_network

        # for bottom most time step
        lls_per_node = np.zeros((data.shape[0], len(template_nodes)))
        unrolled_network_lls_per_node = [lls_per_node]

        # evaluate template bottom up at each time step
        for time_step_num_in_reverse_order in range(total_num_of_time_steps - 1, -1, -1):

            logging.debug(f'time_step_num_in_reverse_order {time_step_num_in_reverse_order}')
            prev_lls_per_node = unrolled_network_lls_per_node[-1]
            logging.debug(f'prev_lls_per_node {prev_lls_per_node}')

            each_time_step_data_for_template = self.get_each_time_step_data_for_template(data,
                                                                                         time_step_num_in_reverse_order,
                                                                                         total_num_of_time_steps,
                                                                                         prev_lls_per_node,
                                                                                         initial_num_latent_interface_nodes,
                                                                                         num_variables_each_time_step,
                                                                                         bottom_up=True)

            if time_step_num_in_reverse_order == 0:

                top_nodes = get_nodes_by_type(self.InitialTemplate.top_network)
                lls_per_node = np.zeros((data.shape[0], len(top_nodes)))
                log_likelihood(self.InitialTemplate.top_network, each_time_step_data_for_template,
                               lls_matrix=lls_per_node)

            else:
                lls_per_node = np.zeros((data.shape[0], len(template_nodes)))
                log_likelihood(self.template, each_time_step_data_for_template, lls_matrix=lls_per_node)

            unrolled_network_lls_per_node.append(lls_per_node)

        print(np.mean(unrolled_network_lls_per_node[-1][:, 0]))

        return unrolled_network_lls_per_node

    def eval_rspmn_top_down(self, data, unrolled_network_lls_per_node):

        logging.debug(f'in method eval_rspmn_top_down()')

        num_variables_each_time_step = len(self.params.feature_names)
        total_num_of_time_steps = int(data.shape[1] / num_variables_each_time_step)
        initial_num_latent_interface_nodes = len(self.InitialTemplate.template_network.children)

        node_functions = get_node_funtions()
        _node_functions_top_down = node_functions[0].copy()
        _node_functions_top_down.update({Max: meu_max})
        logging.debug(f'_node_functions_top_down {_node_functions_top_down}')

        for time_step_num in range(total_num_of_time_steps-1):
            lls_per_node = unrolled_network_lls_per_node[total_num_of_time_steps-time_step_num]

            each_time_step_data_for_template = self.get_each_time_step_data_for_template(data, time_step_num,
                                                                                         total_num_of_time_steps,
                                                                                         lls_per_node,
                                                                                         initial_num_latent_interface_nodes,
                                                                                         num_variables_each_time_step,
                                                                                         bottom_up=False)
            instance_ids = np.arange(each_time_step_data_for_template.shape[0])
            if time_step_num == 0:
                all_results, latent_interface_dict = eval_template_top_down(self.InitialTemplate.top_network,
                                                                            eval_functions=_node_functions_top_down,
                                                                            all_results=None, parent_result=instance_ids,
                                                                            data=each_time_step_data_for_template,
                                                                            lls_per_node=lls_per_node)

            else:

                all_results, latent_interface_dict = eval_template_top_down(self.template,
                                                                            eval_functions=_node_functions_top_down,
                                                                            all_results=None, parent_result=instance_ids,
                                                                            data=each_time_step_data_for_template,
                                                                            lls_per_node=lls_per_node)
            self.template.interface_winner = np.full((each_time_step_data_for_template.shape[0],), np.inf)
            logging.debug(f'latent_interface_dict {latent_interface_dict}')
            for latent_interface_node, instances in latent_interface_dict.items():
                self.template.interface_winner[instances] = latent_interface_node.interface_idx - \
                                                            num_variables_each_time_step

            # if self.template.interface_winner.any(np.inf):
            #     raise Exception(f'All instances are not passed to the corresponding latent interface nodes')


        self.update_weights()

        return unrolled_network_lls_per_node

    @staticmethod
    def get_each_time_step_data_for_template(data, time_step_num, total_num_of_time_steps, lls_per_node,
                                             initial_num_latent_interface_nodes, num_variables_each_time_step,
                                             bottom_up=True):

        each_time_step_data = data[:, (time_step_num * num_variables_each_time_step):
                                      (time_step_num * num_variables_each_time_step) + num_variables_each_time_step]

        assert each_time_step_data.shape[1] == num_variables_each_time_step

        if time_step_num == total_num_of_time_steps - 1:
            # last time step in the sequence. last level of template
            # latent node data is corresponding bottom time step interface node's inference value. 1 for last level
            latent_node_data = np.zeros((each_time_step_data.shape[0], initial_num_latent_interface_nodes))

        else:
            if bottom_up:
                # latent node data is corresponding bottom time step interface node's inference value
                latent_node_data = lls_per_node[:, 1:initial_num_latent_interface_nodes + 1]
            else:
                # latent node data top down is curr time step lls_per_node last set of values
                first_latent_node_column = lls_per_node.shape[1] - initial_num_latent_interface_nodes
                latent_node_data = lls_per_node[:, first_latent_node_column:]

        each_time_step_data_for_template = np.concatenate((each_time_step_data, latent_node_data), axis=1)

        return each_time_step_data_for_template

    def update_weights(self):

        nodes = get_nodes_by_type(self.InitialTemplate.template_network)

        for node in nodes:

            if isinstance(node, Sum):

                if all(isinstance(child, LatentInterface) for child in node.children):

                    for i, child in enumerate(node.children):
                        node.weights[i] = (node.children[i].count/node.count)

                    node.weights = (np.array(node.weights)/np.sum(node.weights)).tolist()

                    print(node.weights)

            print(f'node {node}, count {node.count}')


class RSPMNParams:

    def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names, cluster_by_curr_information_set, util_to_bin):
        self.partial_order = partial_order
        self.decision_nodes = decision_nodes
        self.utility_nodes = utility_nodes
        self.feature_names = feature_names
        self.cluster_by_curr_information_set = cluster_by_curr_information_set
        self.length_of_time_slice = len(self.feature_names)
        self.util_to_bin = util_to_bin

