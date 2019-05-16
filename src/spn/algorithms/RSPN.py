import logging
import numpy as np
import copy
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down, Out_Latent, assign_ids, \
    rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import In_Latent
from spn.algorithms.Inference import log_likelihood

from spn.algorithms.oSLRAU import oSLRAU_eval_spn_top_down, oSLRAU_update_structure, out_latent, oSLRAU_sum

from spn.algorithms.RSPNUtil import initialise_mean_and_covariance

from spn.algorithms.RSPNUtil import oSLRAU_eval_spn_top_down_helper


class RSPN():

    def __init__(self, num_variables, num_latent_variables=2, num_latent_values=2):

        self.num_variables = num_variables
        self.num_latent_variables = num_latent_variables
        self.num_latent_values = num_latent_values
        self.template_spn = None
        self.top_spn = None
        self.len_sequence = None

    def set_len_sequence(self, data):

        assert self.num_variables is not None, "number of variables not specified in RSPN"
        self.len_sequence = int(data.shape[1] / self.num_variables)

    def get_len_sequence(self):
        return self.len_sequence

    def initialise_template(self, initial_spn, initial_data):

        nodes = get_nodes_by_type(initial_spn)
        initial_rows = np.arange(initial_data.shape[0])
        for node in nodes:
            if isinstance(node, Product):
                initialise_mean_and_covariance(node, initial_rows, initial_data )

        latent_val_node_list = []
        for i in range(self.num_latent_values):
            spn_i = copy.deepcopy(initial_spn)
            latent_val_node_list.append(spn_i)

        # in_latent_children = []
        scope = [self.num_variables + i for i in range(self.num_latent_variables)]
        # for i in range(self.num_latent_variables):
        #     in_latent_i = In_Latent(bin_value=0, scope=scope)
        #     in_latent_children.append(in_latent_i)

        interface_node_list = []
        for i in range(self.num_latent_variables):

            latent_val_node_list_copy = []
            for j, spn in enumerate(latent_val_node_list):
                weights = np.random.random_sample(self.num_latent_variables)
                weights = list(weights / np.sum(weights))
                in_latent_children = []
                for i in range(self.num_latent_variables):
                    in_latent_i = In_Latent(bin_value=0, interface_index=i, scope=scope)
                    in_latent_children.append(in_latent_i)
                s_j = Sum(weights=weights, children=in_latent_children)
                s_j.count = 1

                if isinstance(spn, Product):
                    spn1 = copy.deepcopy(spn)
                    spn1.children.extend([s_j])
                    rebuild_scopes_bottom_up(spn1)

                elif isinstance(spn, Sum):
                    spn1 = copy.deepcopy(spn)
                    spn1 = Product(children=[spn1, s_j])
                    rebuild_scopes_bottom_up(spn1)
                    initialise_mean_and_covariance(spn1, initial_rows, initial_data)

                else:
                    spn1 = copy.deepcopy(spn)
                    spn1 = Product(children=[spn1, s_j])
                    rebuild_scopes_bottom_up(spn1)
                    initialise_mean_and_covariance(spn1, initial_rows, initial_data)

                latent_val_node_list_copy.append(spn1)

            weights = np.random.random_sample(self.num_latent_values)
            weights = list(weights / np.sum(weights))
            interface_spn_i = Sum(weights=weights, children=latent_val_node_list_copy)
            # interface_spn_i = copy.deepcopy(spn)
            interface_node_list.append(interface_spn_i)

        self.template_spn = Out_Latent(children=interface_node_list)
        assign_ids(self.template_spn)
        rebuild_scopes_bottom_up(self.template_spn)

        return self.template_spn

    def initialise_top_spn(self):

        in_latent_children = []
        weights = np.random.random_sample(self.num_latent_variables)
        weights = list(weights / np.sum(weights))

        scope = [self.num_variables + i for i in range(self.num_latent_variables)]

        for i in range(self.num_latent_variables):
            in_latent_i = In_Latent(bin_value=0, interface_index=i, scope=scope)
            in_latent_children.append(in_latent_i)

        self.top_spn = Sum(weights=weights, children=in_latent_children)
        assign_ids(self.top_spn)
        rebuild_scopes_bottom_up(self.top_spn)

        return self.top_spn

    def evaluate_top_spn_bottom_up(self, data, unrolled_network_lls_per_node):

        nodes = get_nodes_by_type(self.top_spn)
        lls_per_node = unrolled_network_lls_per_node[self.len_sequence-1].copy()
        dummy_data = np.zeros((data.shape[0], self.num_latent_variables))

        # for k, node in enumerate(nodes):
        for node in nodes:
            if isinstance(node, In_Latent):
                k = node.interface_index
                node.inference_value = np.exp(lls_per_node[:, 1 + k]).reshape(-1, 1)

        lls_per_node = np.zeros((data.shape[0], len(nodes)))
        log_likelihood(self.top_spn, dummy_data, lls_matrix=lls_per_node)

        return lls_per_node

    def evaluate_top_spn_top_down(self, data, lls_per_node, oSLRAU_params):

        dummy_data = np.zeros((data.shape[0], self.num_latent_variables))
        instance_ids = np.arange(dummy_data.shape[0])
        self.template_spn.out_latent_winner = np.zeros((data.shape[0],), dtype=int)

        all_results, nodes_to_update, in_latent_dict = oSLRAU_eval_spn_top_down(self.top_spn, oSLRAU_params,
                                                                                parent_result=instance_ids,
                                                                                data=dummy_data,
                                                                                lls_per_node=lls_per_node)
        #i = 0
        for in_latent_node, instances in in_latent_dict.items():
            #interface_node_index = in_latent_node.scope[i] - self.num_variables
            self.template_spn.out_latent_winner[instances] = in_latent_node.interface_index
            #i = i+1

        return

    def evaluate_rspn(self, data, oSLRAU_params, unroll, full_update):

        assert unroll == 'backward' or unroll == 'forward'; "specify unroll - forward or backward"
        assert data.shape[1] == self.num_variables*self.len_sequence, "data columns not equal to number of variables time length of sequence"
        nodes = get_nodes_by_type(self.template_spn)
        # lls_per_node = np.zeros((data.shape[0], len(nodes)))

        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        for index in in_latent_index:
            k = nodes[index].interface_index
            nodes[index].inference_value = 1

        unrolled_network_lls_per_node = self.evaluate_rspn_bottom_up(data, unroll)

        top_spn_lls_per_node = self.evaluate_top_spn_bottom_up(data, unrolled_network_lls_per_node)
        self.evaluate_top_spn_top_down(data, top_spn_lls_per_node, oSLRAU_params)

        nodes_to_update = self.evaluate_rspn_top_down(data, unrolled_network_lls_per_node,
                                                      oSLRAU_params, unroll, full_update)
        return nodes_to_update

    def evaluate_rspn_bottom_up(self, data, unroll):

        nodes = get_nodes_by_type(self.template_spn)
        # lls_per_node = np.zeros((data.shape[0], len(nodes)))

        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        unrolled_network_lls_per_node = []
        for i in range(self.len_sequence):

            lls_per_node = np.zeros((data.shape[0], len(nodes)))
            if unroll == 'backward':
                j = self.len_sequence - i -1
                data_i = data[:, (j * self.num_variables): (j * self.num_variables) + (self.num_variables)]
            else:
                data_i = data[:, (i*self.num_variables) : (i*self.num_variables) + (self.num_variables)]

            assert data_i.shape[1] == self.num_variables
            log_likelihood(self.template_spn, data_i, lls_matrix=lls_per_node)
            unrolled_network_lls_per_node.append(lls_per_node)
            # for k, index in enumerate(in_latent_index):
            for index in in_latent_index:
                k = nodes[index].interface_index
                nodes[index].inference_value = np.exp(lls_per_node[:, 1 + k]).reshape(-1, 1)

        return unrolled_network_lls_per_node

    def evaluate_rspn_top_down(self, data, unrolled_network_lls_per_node, oSLRAU_params, unroll, full_update):

        # l = self.len_sequence
        # num_variables = data.shape[1]
        # self.template_spn.out_latent_winner = np.zeros((data.shape[0],), dtype=int)
        for i in range(self.len_sequence):

            instance_ids = np.arange(data.shape[0])
            if unroll == 'backward':
                data_i = data[:, (i*self.num_variables) : (i*self.num_variables) + (self.num_variables)]
                # lls_per_node = unrolled_network_lls_per_node[i]
            else :
                j = self.len_sequence - i - 1
                data_i = data[:, (j * self.num_variables): (j * self.num_variables) + (self.num_variables)]


            assert data_i.shape[1] == self.num_variables
            lls_per_node = unrolled_network_lls_per_node[self.len_sequence - i - 1]

            if full_update == False:
                if i == 0:

                    all_results, nodes_to_update, in_latent_dict = oSLRAU_eval_spn_top_down(self.template_spn, oSLRAU_params,
                                                                                            parent_result=instance_ids,
                                                                                            data=data_i,
                                                                                            lls_per_node=lls_per_node)
                else:

                    parent_result = [np.array(instance_ids)]
                    children_row_ids = out_latent(self.template_spn, parent_result)

                    for node, param in children_row_ids.items():
                        grand_children_row_ids = oSLRAU_sum(node, [param], data=data_i, lls_per_node=lls_per_node)

                        for node_1, param_1 in grand_children_row_ids.items():
                            all_results, nodes_to_update, in_latent_dict = oSLRAU_eval_spn_top_down_helper(node_1,
                                                                                                    oSLRAU_params,
                                                                                                    parent_result=param_1,
                                                                                                    data=data_i,
                                                                                                    lls_per_node=lls_per_node)


                            for in_latent_node, instances in in_latent_dict.items():

                                self.template_spn.out_latent_winner[instances] = in_latent_node.interface_index



            else:
                all_results, nodes_to_update, in_latent_dict = oSLRAU_eval_spn_top_down(self.template_spn,
                                                                                        oSLRAU_params,
                                                                                        parent_result=instance_ids,
                                                                                        data=data_i,
                                                                                        lls_per_node=lls_per_node)

                for in_latent_node, instances in in_latent_dict.items():
                    # interface_node_index = in_latent_node.scope[i] - self.num_variables
                    self.template_spn.out_latent_winner[instances] = in_latent_node.interface_index




        return nodes_to_update

    def update_template_spn(self, oSLRAU_params, nodes_to_update):
        nodes = get_nodes_by_type(self.template_spn)
        tot_nodes_len = len(nodes)
        self.template_spn = oSLRAU_update_structure(self.template_spn, oSLRAU_params, tot_nodes_len, nodes_to_update)

        return self.template_spn

    def rspn_log_likelihood(self, data, unroll):

        assert data.shape[1] == self.num_variables*self.len_sequence, "data columns not equal to number of variables times length of sequence"
        nodes = get_nodes_by_type(self.template_spn)
        # lls_per_node = np.zeros((data.shape[0], len(nodes)))

        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        for index in in_latent_index:
            k = nodes[index].interface_index
            nodes[index].inference_value = 1

        unrolled_network_lls_per_node = self.evaluate_rspn_bottom_up(data, unroll)

        top_spn_lls_per_node = self.evaluate_top_spn_bottom_up(data, unrolled_network_lls_per_node)

        return top_spn_lls_per_node[:, 0]