import logging
import numpy as np
import copy
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down, In_Latent, Out_Latent, assign_ids, rebuild_scopes_bottom_up

def initialise_template(initial_spn, num_variables, num_latent_variables, num_latent_values=2):

    latent_val_node_list = []
    for i in range(num_latent_values):
        spn_i = copy.deepcopy(initial_spn)
        latent_val_node_list.append(spn_i)

    children = []
    for i in range(num_latent_variables):
        in_latent_i = In_Latent(bin_value=0, scope=num_variables+i)
        children.append(in_latent_i)

    latent_val_node_list_copy = [None, None]
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


    # assign_ids(spn)
    # rebuild_scopes_bottom_up(spn)
    #latent_val_node_list_copy = [None, None]#copy.deepcopy(latent_val_node_list)
    interface_node_list = []
    for i in range(num_latent_variables):

        for j, spn in enumerate(latent_val_node_list):
            weights = np.random.random_sample(num_latent_variables)
            weights = list(weights / np.sum(weights))
            s_j = Sum(weights=weights, children=children)
            s_j.count = 1

            if isinstance(spn, Product):
                spn1 = copy.deepcopy(spn)
                spn1.children.extend([s_j])

            elif isinstance(spn, Sum):
                spn1 = copy.copy(spn)
                spn1 = Product(children=[spn1, s_j])

            latent_val_node_list_copy[j] = spn1

        weights = np.random.random_sample(num_latent_values)
        weights = list(weights / np.sum(weights))
        spn = Sum(weights=weights, children=latent_val_node_list_copy)
        interface_spn_i = copy.deepcopy(spn)
        interface_node_list.append(interface_spn_i)

    dummy_root = Out_Latent(children = interface_node_list)
    assign_ids(dummy_root)
    rebuild_scopes_bottom_up(dummy_root)
    return dummy_root