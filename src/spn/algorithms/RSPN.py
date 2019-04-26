import logging
import numpy as np
import copy
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down, Out_Latent, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import In_Latent
from spn.algorithms.Inference import log_likelihood

from spn.algorithms.oSLRAU import oSLRAU_eval_spn_top_down


class RSPN():

    def __init__(self, num_variables, num_latent_variables=2, num_latent_values=2):

        self.num_variables = num_variables
        self.num_latent_variables = num_latent_variables
        self.num_latent_values = num_latent_values
        self.template_spn = None
        self.top_spn = None



    def initialise_template(self, initial_spn):

        latent_val_node_list = []
        for i in range(self.num_latent_values):
            spn_i = copy.deepcopy(initial_spn)
            latent_val_node_list.append(spn_i)

        in_latent_children = []
        for i in range(self.num_latent_variables):
            in_latent_i = In_Latent(bin_value=0, scope=self.num_variables+i)
            in_latent_children.append(in_latent_i)

        interface_node_list = []
        for i in range(self.num_latent_variables):

            latent_val_node_list_copy = []
            for j, spn in enumerate(latent_val_node_list):
                weights = np.random.random_sample(self.num_latent_variables)
                weights = list(weights / np.sum(weights))
                s_j = Sum(weights=weights, children=in_latent_children)
                s_j.count = 1

                if isinstance(spn, Product):
                    spn1 = copy.deepcopy(spn)
                    spn1.children.extend([s_j])

                elif isinstance(spn, Sum):
                    spn1 = copy.copy(spn)
                    spn1 = Product(children=[spn1, s_j])

                latent_val_node_list_copy.append(spn1)

            weights = np.random.random_sample(self.num_latent_values)
            weights = list(weights / np.sum(weights))
            interface_spn_i = Sum(weights=weights, children=latent_val_node_list_copy)
            # interface_spn_i = copy.deepcopy(spn)
            interface_node_list.append(interface_spn_i)

        self.template_spn = Out_Latent(children = interface_node_list)
        assign_ids(self.template_spn)
        rebuild_scopes_bottom_up(self.template_spn)
        return self.template_spn


    def initialise_top_spn(self):

        in_latent_children = []
        weights = np.random.random_sample(self.num_latent_variables)
        weights = list(weights / np.sum(weights))

        for i in range(self.num_latent_variables):
            in_latent_i = In_Latent(bin_value=0, scope=self.num_variables + i)
            in_latent_children.append(in_latent_i)

        self.top_spn = Sum(weights=weights, children=in_latent_children)

        return self.top_spn


    def evaluate_rspn(self, data, len_sequence, oSLRAU_params):

        unrolled_network_lls_per_node = self.evaluate_rspn_bottom_up(self.template_spn, data, len_sequence)
        nodes_to_update = self.evaluate_rspn_top_down(self.template_spn, data, len_sequence, unrolled_network_lls_per_node,
                                                 oSLRAU_params)
        return nodes_to_update


    def evaluate_rspn_bottom_up(self, data, len_sequence):

        nodes = get_nodes_by_type(self.template_spn)
        #lls_per_node = np.zeros((data.shape[0], len(nodes)))

        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        unrolled_network_lls_per_node = []
        for i in range(len_sequence):
            lls_per_node = np.zeros((data.shape[0], len(nodes)))
            log_likelihood(self.template_spn, data, lls_matrix=lls_per_node)
            unrolled_network_lls_per_node.append(lls_per_node)
            for k, index in enumerate(in_latent_index):
                nodes[index].inference_value = np.exp(lls_per_node[:, 1+k]).reshape(-1, 1)

        return unrolled_network_lls_per_node


    def evaluate_rspn_top_down(self, data, len_sequence, unrolled_network_lls_per_node, oSLRAU_params):

        l = len_sequence
        num_variables = data.shape[1]
        self.template_spn.out_latent_winner = np.zeros((data.shape[0],), dtype=int)
        for i in range(l):

            instance_ids = np.arange(data.shape[0])
            data = data
            lls_per_node = unrolled_network_lls_per_node[l-i-1]
            all_results, nodes_to_update, in_latent_dict = oSLRAU_eval_spn_top_down(template_spn, oSLRAU_params, parent_result=instance_ids,
                                     data=data, lls_per_node=lls_per_node)

            for in_latent_node, instances in in_latent_dict.items():
                interface_node_index = in_latent_node.scope[0] - num_variables
                self.template_spn.out_latent_winner[instances] = interface_node_index

            return nodes_to_update
























