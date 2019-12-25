import collections
import copy
import logging

from spn.algorithms.EM import get_node_updates_for_EM
#from spn.algorithms.Gradient import gradient_backward
from spn.structure.Base import Leaf, Sum, InterfaceSwitch, assign_ids

from spn.algorithms.SPMN import SPMN, SPMNParams

from spn.algorithms.RSPMNInitialTemplateBuild import RSPMNInitialTemplate

from spn.algorithms.Inference import log_likelihood
from spn.structure.Base import get_nodes_by_type
import numpy as np

from spn.algorithms.MPE import get_node_funtions, mpe
from spn.algorithms.RSMPN.RSPMNUtil import eval_template_top_down, gradient_backward

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
    def pass_meu_val_to_latent_interface_leaf_nodes(
            eval_val_per_node, prev_eval_val_per_node,
            initial_num_latent_interface_nodes
            ):

        # leaf latent node columns in eval_val_per_node correspond to
        # last few columns in current time step. They are equal to
        # num of latent interface nodes
        first_latent_node_column = \
            eval_val_per_node.shape[1] - \
            initial_num_latent_interface_nodes

        # eval val of last columns corr to latent interface nodes of
        # current time step are equal to first columns corr to
        # eval vals of prev time step
        eval_val_per_node[:, first_latent_node_column:] = \
            prev_eval_val_per_node[:, 1:initial_num_latent_interface_nodes + 1]

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

    def meu(self, template, data):

        unrolled_network_meu_per_node = self.eval_rspmn_bottom_up(template,
                                                                  data, False)
        # ll at root node
        meu = unrolled_network_meu_per_node[-1][:, 0]

        return meu, unrolled_network_meu_per_node

    def EM_optimization(self, template, data, iterations=5,
                        skip_validation=False, **kwargs):
        # if not skip_validation:
        #     valid, err = is_valid(spn)
        #     assert valid, "invalid spn: " + err

        node_updates = get_node_updates_for_EM()

        # lls_per_node = np.zeros((data.shape[0], get_number_of_nodes(template)))

        for _ in range(iterations):
            # one pass bottom up evaluating the likelihoods
            ll, unrolled_network_lls_per_node = self.log_likelihood(template, data)

            template = self.rspmn_gradient_backward(template, data,
                                                    unrolled_network_lls_per_node,
                                                    node_updates, **kwargs)

            self.template = template

            return template

    def rspmn_gradient_backward(self, template, data,
                                unrolled_network_lls_per_node, node_updates,
                                **kwargs):

        num_variables_each_time_step, total_num_of_time_steps, \
            initial_num_latent_interface_nodes = \
            self.get_params_for_get_each_time_step_data_for_template(template,
                                                                     data)

        latent_queue = collections.deque()
        for time_step_num in range(total_num_of_time_steps-1):
            lls_per_node = unrolled_network_lls_per_node[
                total_num_of_time_steps - time_step_num
                ]

            each_time_step_data_for_template = \
                self.get_each_time_step_data_for_template(
                    data, time_step_num,
                    total_num_of_time_steps,
                    lls_per_node,
                    initial_num_latent_interface_nodes,
                    num_variables_each_time_step,
                    bottom_up=False
                )
            node_gradients = get_node_gradients()[0].copy()

            if time_step_num == 0:

                R = lls_per_node[:, 0]
                gradients, latent_interface_dict = \
                    gradient_backward(self.InitialTemplate.top_network,
                                      lls_per_node, node_gradients,
                                      data=each_time_step_data_for_template)

                next_time_step_latent_queue = collections.deque()
                top_down_pass_val_dict = {}
                i = 0
                for latent_interface_node, top_down_pass_val in \
                        latent_interface_dict.items():

                    template_child_num = i % len(template.children)
                    template_child = template.children[template_child_num]
                    top_down_pass_val_dict[template_child] = \
                        top_down_pass_val

                    if (i + 1) % len(template.children) == 0:
                        next_time_step_latent_queue.append(
                            top_down_pass_val_dict
                        )
                        top_down_pass_val_dict = {}
                    i += 1

                for node_type, func in node_updates.items():
                    for node in get_nodes_by_type(self.InitialTemplate.top_network, node_type):
                        func(
                            node,
                            node_lls=lls_per_node[:, node.id],
                            node_gradients=gradients[:, node.id],
                            root_lls=R,
                            all_lls=lls_per_node,
                            all_gradients=gradients,
                            data=each_time_step_data_for_template,
                            **kwargs
                        )

            else:

                R = lls_per_node[:, 0]
                next_time_step_latent_queue = collections.deque()
                while latent_queue:
                    print(f'length of latent queue {len(latent_queue)}')
                    template.top_down_pass_val = latent_queue.popleft()
                    gradients, latent_interface_dict = \
                        gradient_backward(template,
                                          lls_per_node, node_gradients,
                                          data=each_time_step_data_for_template)

                    top_down_pass_val_dict = {}
                    i = 0
                    for latent_interface_node, top_down_pass_val in \
                            latent_interface_dict.items():

                        template_child_num = i % len(template.children)
                        template_child = template.children[template_child_num]
                        top_down_pass_val_dict[template_child] = \
                            top_down_pass_val

                        if (i+1) % len(template.children) == 0:
                            next_time_step_latent_queue.append(
                                top_down_pass_val_dict
                            )
                            top_down_pass_val_dict = {}
                        i += 1

                    for node_type, func in node_updates.items():
                        for node in get_nodes_by_type(template, node_type):
                            func(
                                node,
                                node_lls=lls_per_node[:, node.id],
                                node_gradients=gradients[:, node.id],
                                root_lls=R,
                                all_lls=lls_per_node,
                                all_gradients=gradients,
                                data=data,
                                **kwargs
                            )
            logging.debug(f'latent_interface_dict {latent_interface_dict}')

            latent_queue = next_time_step_latent_queue

            # if self.template.interface_winner.any(np.inf):
            #     raise Exception(f'All instances are not passed to
            #     the corresponding latent interface nodes')
        return template

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
