import logging
import numpy as np
import copy
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down, Out_Latent, assign_ids, \
    rebuild_scopes_bottom_up, Leaf
from spn.structure.leaves.parametric.Parametric import In_Latent
from spn.algorithms.Inference import log_likelihood

from spn.algorithms.oSLRAU import oSLRAU_eval_spn_top_down, oSLRAU_update_structure, out_latent, oSLRAU_sum

from spn.algorithms.RSPNUtil import initialise_mean_and_covariance

from spn.algorithms.RSPNUtil import eval_rspn_top_down_partial_update

from spn.io.Graphics import plot_spn

from spn.algorithms.TransformStructure import Prune

from spn.algorithms.LearningWrappers import learn_parametric


class RSPN():

    def __init__(self, num_variables, num_latent_variables=2, num_latent_values=2):

        self.num_variables = num_variables
        self.num_latent_variables = num_latent_variables
        self.num_latent_values = num_latent_values
        self.template_spn = None
        self.top_spn = None
        self.len_sequence = None

    def build_initial_template(self, mini_batch, ds_context, len_sequence_varies=False):

        if len_sequence_varies:
            assert type(mini_batch) is list, 'When sequence length varies, data is a list of numpy arrays'
        else:
            assert type(mini_batch) is np.ndarray, 'data should be of type numpy array'

        print("Building initial spn")

        learn_spn_data = self.__get_learn_spn_data(mini_batch, len_sequence_varies)

        spn = learn_parametric(learn_spn_data, ds_context)
        # print("initial spn", spn)

        print("Building initial template spn")
        initial_template_spn = copy.deepcopy(self.initialise_template(spn, learn_spn_data))

        print("Building top spn")
        top_spn = self.initialise_top_spn()

        return spn, initial_template_spn, top_spn

    def __get_learn_spn_data(self, mini_batch, len_sequence_varies):

        if len_sequence_varies:
            for i in range(len(mini_batch)):
                mini_batch_data = mini_batch[i]
                learn_spn_data_i = mini_batch_data[:, 0:self.num_variables]
                if i == 0:
                    learn_spn_data = learn_spn_data_i
                else:
                    learn_spn_data = np.concatenate((learn_spn_data, learn_spn_data_i))
        else:
            learn_spn_data = mini_batch[:, 0:self.num_variables]

        return learn_spn_data

    def learn_rspn(self, mini_batch, update_template, oSLRAU_params, unroll, full_update, update_leaves,
                   len_sequence_varies=False):

        if len_sequence_varies:
            assert type(mini_batch) is list, 'When sequence length varies, data is a list of numpy arrays'
            print("Evaluating rspn and collecting nodes to update")

            for i in range(len(mini_batch)):
                mini_batch_data = mini_batch[i]
                self.set_len_sequence(mini_batch_data)
                # print("length of sequence:", self.get_len_sequence())

                nodes_to_update = self.evaluate_rspn(mini_batch_data, oSLRAU_params, unroll, full_update,
                                                     update_leaves)
        else:
            assert type(mini_batch) is np.ndarray, 'data should be of type numpy array'

            self.set_len_sequence(mini_batch)
            len_seq = self.get_len_sequence()

            # print("Length of the sequence in mini_batch:", len_seq)
            assert mini_batch.shape[
                       1] == self.num_variables * self.len_sequence, "data columns not equal to number of variables time length of sequence"
            # print("Evaluating rspn and collecting nodes to update")

            nodes_to_update = self.evaluate_rspn(mini_batch, oSLRAU_params, unroll, full_update,
                                                 update_leaves)

        if update_template:
            print("Updating template spn")
            self.update_template_spn(oSLRAU_params, nodes_to_update)

        return self.template_spn

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
                initialise_mean_and_covariance(node, initial_rows, initial_data)

        latent_val_node_list = []
        for i in range(self.num_latent_values):
            spn_i = copy.deepcopy(initial_spn)
            latent_val_node_list.append(spn_i)

        scope = [self.num_variables + i for i in range(self.num_latent_variables)]

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
        lls_per_node = unrolled_network_lls_per_node[self.len_sequence - 1].copy()
        dummy_data = np.zeros((data.shape[0], self.num_latent_variables))

        for node in nodes:
            if isinstance(node, In_Latent):
                k = node.interface_index
                node.log_inference_value = lls_per_node[:, 1 + k].reshape(-1, 1)

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

        for in_latent_node, instances in in_latent_dict.items():
            self.template_spn.out_latent_winner[instances] = in_latent_node.interface_index

        return

    def evaluate_rspn(self, data, oSLRAU_params, unroll, full_update, update_leaves=None):

        assert unroll == 'backward' or unroll == 'forward';
        "specify unroll as forward or backward"
        assert data.shape[
                   1] == self.num_variables * self.len_sequence, "data columns not equal to number of variables time length of sequence"
        assert full_update == True or full_update == False, "Specify full_update as True or False"
        if full_update == False:
            assert update_leaves == True or update_leaves == False, "For partial update, specify whether update_leaves is True or False"
        else:
            assert update_leaves == None, "Leaves are always updated when full_update is True, specify update_leaves as None"

        nodes = get_nodes_by_type(self.template_spn)
        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        for index in in_latent_index:
            nodes[index].log_inference_value = 0

        unrolled_network_lls_per_node = self.evaluate_rspn_bottom_up(data, unroll)

        top_spn_lls_per_node = self.evaluate_top_spn_bottom_up(data, unrolled_network_lls_per_node)
        self.evaluate_top_spn_top_down(data, top_spn_lls_per_node, oSLRAU_params)

        nodes_to_update = self.evaluate_rspn_top_down(data, unrolled_network_lls_per_node,
                                                      oSLRAU_params, unroll, full_update, update_leaves)
        return nodes_to_update

    def evaluate_rspn_bottom_up(self, data, unroll):

        nodes = get_nodes_by_type(self.template_spn)
        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        unrolled_network_lls_per_node = []
        for i in range(self.len_sequence):

            lls_per_node = np.zeros((data.shape[0], len(nodes)))
            if unroll == 'backward':
                j = self.len_sequence - i - 1
                data_i = data[:, (j * self.num_variables): (j * self.num_variables) + (self.num_variables)]
            else:
                data_i = data[:, (i * self.num_variables): (i * self.num_variables) + (self.num_variables)]

            assert data_i.shape[1] == self.num_variables

            log_likelihood(self.template_spn, data_i, lls_matrix=lls_per_node)
            unrolled_network_lls_per_node.append(lls_per_node)

            for index in in_latent_index:
                k = nodes[index].interface_index
                nodes[index].log_inference_value = lls_per_node[:, 1 + k].reshape(-1, 1)

        return unrolled_network_lls_per_node

    def evaluate_rspn_top_down(self, data, unrolled_network_lls_per_node, oSLRAU_params, unroll, full_update,
                               update_leaves):

        for i in range(self.len_sequence):

            instance_ids = np.arange(data.shape[0])
            if unroll == 'backward':
                data_i = data[:, (i * self.num_variables): (i * self.num_variables) + (self.num_variables)]

            else:
                j = self.len_sequence - i - 1
                data_i = data[:, (j * self.num_variables): (j * self.num_variables) + (self.num_variables)]

            assert data_i.shape[1] == self.num_variables
            lls_per_node = unrolled_network_lls_per_node[self.len_sequence - i - 1]

            if full_update == False:
                if i == 0:
                    all_results, nodes_to_update, in_latent_dict = oSLRAU_eval_spn_top_down(self.template_spn,
                                                                                            oSLRAU_params,
                                                                                            parent_result=instance_ids,
                                                                                            data=data_i,
                                                                                            lls_per_node=lls_per_node)
                else:
                    parent_result = [np.array(instance_ids)]
                    children_row_ids = out_latent(self.template_spn, parent_result)

                    for child, child_param in children_row_ids.items():
                        grand_children_row_ids = oSLRAU_sum(child, [child_param], data=data_i,
                                                            lls_per_node=lls_per_node)

                        for grand_child, grand_child_param in grand_children_row_ids.items():
                            all_results, in_latent_dict = eval_rspn_top_down_partial_update(grand_child,
                                                                                            oSLRAU_params,
                                                                                            update_leaves,
                                                                                            parent_result=grand_child_param,
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
                    self.template_spn.out_latent_winner[instances] = in_latent_node.interface_index

        return nodes_to_update

    def update_template_spn(self, oSLRAU_params, nodes_to_update):

        nodes = get_nodes_by_type(self.template_spn)
        tot_nodes_len = len(nodes)
        self.template_spn = oSLRAU_update_structure(self.template_spn, oSLRAU_params, tot_nodes_len, nodes_to_update)

        return self.template_spn

    def log_likelihood(self, data, unroll, len_sequence_varies):

        nodes = get_nodes_by_type(self.template_spn)
        in_latent_index = []
        for j in range(len(nodes)):
            if isinstance(nodes[j], In_Latent):
                in_latent_index.append(j)

        if len_sequence_varies:
            assert type(data) is list, 'When sequence length varies, data is a list of numpy arrays'
            print("Evaluating rspn bottom up")
            for i in range(len(data)):
                for index in in_latent_index:
                    nodes[index].log_inference_value = 0

                each_data_point = data[i]
                self.set_len_sequence(each_data_point)

                # print("length of sequence:", self.get_len_sequence())


                unrolled_network_lls_per_node = self.evaluate_rspn_bottom_up(each_data_point, unroll)
                top_spn_lls_per_node = self.evaluate_top_spn_bottom_up(each_data_point, unrolled_network_lls_per_node)

                if i == 0:
                    ll = top_spn_lls_per_node[:, 0]
                else:
                    ll = np.concatenate((ll, top_spn_lls_per_node[:, 0]))
        else:
            assert type(data) is np.ndarray, 'data should be of type numpy array'
            for index in in_latent_index:
                nodes[index].log_inference_value = 0

            self.set_len_sequence(data)
            len_seq = self.get_len_sequence()

            print("Length of the sequence in mini_batch:", len_seq)
            assert data.shape[
                       1] == self.num_variables * self.len_sequence, "data columns not equal to number of variables times length of sequence"
            print("Evaluating rspn bottom up")

            unrolled_network_lls_per_node = self.evaluate_rspn_bottom_up(data, unroll)

            top_spn_lls_per_node = self.evaluate_top_spn_bottom_up(data, unrolled_network_lls_per_node)

            ll = top_spn_lls_per_node[:, 0]

        return ll

    def get_unrolled_rspn(self, len_sequence):

        unroll_spn = []
        for i in range(len_sequence):
            spn_i = copy.deepcopy(self.template_spn)
            unroll_spn.append(spn_i)

        top_spn = copy.deepcopy(self.top_spn)
        unroll_spn.append(top_spn)

        for i in range(len(unroll_spn)):
            spn = unroll_spn[len_sequence - i]
            nodes = get_nodes_by_type(spn)
            for node in nodes:
                if not isinstance(node, Leaf):
                    node_children = node.children.copy()
                    for child in node_children:
                        if type(child) == In_Latent:
                            if len_sequence - i > 0:
                                below_network_out_latent = unroll_spn[(len_sequence - i) - 1]
                                below_interface_node = below_network_out_latent.children[child.interface_index]
                                node.children.extend([below_interface_node])
                                node.children.remove(child)
                            else:
                                node.children.remove(child)

        nodes = get_nodes_by_type(top_spn)
        for node in nodes:
            if type(node) == Out_Latent:
                del node
            if not isinstance(node, Leaf):
                for child in node.children:
                    if not isinstance(child, Leaf):
                        if len(child.children) == 0:
                            node.children.remove(child)
                            del child

        assign_ids(top_spn)
        rebuild_scopes_bottom_up(top_spn)
        # Prune(top_spn)
        return top_spn
