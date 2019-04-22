import logging
import numpy as np
import copy
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down, Out_Latent, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import In_Latent
from spn.algorithms.Inference import log_likelihood


def initialise_template(initial_spn, num_variables, num_latent_variables, num_latent_values=2):

    latent_val_node_list = []
    for i in range(num_latent_values):
        spn_i = copy.deepcopy(initial_spn)
        latent_val_node_list.append(spn_i)

    in_latent_children = []
    for i in range(num_latent_variables):
        in_latent_i = In_Latent(bin_value=0, scope=num_variables+i)
        in_latent_children.append(in_latent_i)

    #
    # for j, spn in enumerate(latent_val_node_list):
    #     weights = np.random.random_sample(num_latent_variables)
    #     weights = list(weights / np.sum(weights))
    #     s_j = Sum(weights=weights, children=children)
    #     s_j.count = 1
    #
    #     if isinstance(spn, Product):
    #         spn1 = copy.copy(spn)
    #         spn1.children.extend([s_j])
    #
    #     elif isinstance(spn, Sum):
    #         spn1 = copy.copy(spn)
    #         spn1 = Product(children=[spn1, s_j])
    #
    #     latent_val_node_list_copy[j] = spn1

    #
    # assign_ids(spn)
    # rebuild_scopes_bottom_up(spn)
    # latent_val_node_list_copy = [None, None]#copy.deepcopy(latent_val_node_list)

    interface_node_list = []
    for i in range(num_latent_variables):

        latent_val_node_list_copy = []
        for j, spn in enumerate(latent_val_node_list):
            weights = np.random.random_sample(num_latent_variables)
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

        weights = np.random.random_sample(num_latent_values)
        weights = list(weights / np.sum(weights))
        interface_spn_i = Sum(weights=weights, children=latent_val_node_list_copy)
        #interface_spn_i = copy.deepcopy(spn)
        interface_node_list.append(interface_spn_i)

    dummy_root = Out_Latent(children = interface_node_list)
    assign_ids(dummy_root)
    rebuild_scopes_bottom_up(dummy_root)
    return dummy_root

def evaluate_rspn(template_spn, data, len_sequence):

    nodes = get_nodes_by_type(template_spn)
    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    in_latent_index = []
    for j in range(len(nodes)):
        if isinstance(nodes[j], In_Latent):
            in_latent_index.append(j)

    unrolled_network_lls_per_node = []
    for i in range(len_sequence):
        lls_per_node = np.zeros((data.shape[0], len(nodes)))
        log_likelihood(template_spn, data, lls_matrix=lls_per_node)
        unrolled_network_lls_per_node.append(lls_per_node)
        for k, index in enumerate(in_latent_index):
            nodes[index].inference_value = np.exp(lls_per_node[:, 1+k]).reshape(-1, 1)


