import collections
import copy
import sys
print(sys.path)
# sys.path.append('/home/hari/Desktop/Projects/Thesis_Project/SPFlow_clone/SPFlow/src')
import numpy as np
import logging
from spn.structure.Base import Leaf, Product, Sum, LatentInterface, InterfaceSwitch, assign_ids

from spn.algorithms.SPMN import SPMN, SPMNParams

from spn.algorithms.RSPMNHelper import get_partial_order_two_time_steps, get_feature_names_two_time_steps \
    , get_nodes_two_time_steps


class RSPMN:

    def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names,
                 cluster_by_curr_information_set=False, util_to_bin=False):

        self.params = SPMNParams(partial_order, decision_nodes, utility_nodes, feature_names, util_to_bin)
        self.two_time_step_params = RSPMNTwoTimeStepParams(self.params)
        self.length_of_time_slice = len(self.params.feature_names)
        self.op = 'Any'
        self.cluster_by_curr_information_set = cluster_by_curr_information_set
        self.spmn_structure_two_time_steps = None

    def build_initial_template(self, spmn_for_two_time_slices):

        logging.debug(f'length_of_time_slice {self.length_of_time_slice}')

        # scope of each random variable in the SPMN is its position in feature_names which follows partial order
        # top interface parent's scope is utility0 scope plus scope of time slice 1
        scope_top_interface_parent = [scope for scope, var in
                                      enumerate(self.two_time_step_params.feature_names_two_time_steps)
                                      if var not in self.two_time_step_params.decision_nodes_two_time_steps
                                      and scope >= self.length_of_time_slice-1]

        logging.debug(f'scope_top_interface_parent {scope_top_interface_parent}')

        top_layer, interface_layer = self.get_top_and_interface_layers(spmn_for_two_time_slices,
                                                            scope_top_interface_parent)

        return top_layer, interface_layer

    def get_top_and_interface_layers(self, root, scope_top_interface_parent):

        interface_nodes = []
        scope_time_slice_1 = scope_top_interface_parent[1:]
        print(f'scope_time_slice_1 {scope_time_slice_1}')

        # perform bfs
        seen, queue = set([root]), collections.deque([root])
        interface_idx = 0
        while queue:
            node = queue.popleft()
            if not isinstance(node, Leaf):

                time_step_1_children = []

                for child in node.children:
                    # print(f'node.scope {node.scope}')

                    # check for top interface parent node
                    if isinstance(node, Product) and node.scope == scope_top_interface_parent:

                        print(f'scope_top_interface_parent {node.scope}')
                        print(f'child.scope {child.scope}')
                        if set(child.scope).issubset(scope_time_slice_1):
                            time_step_1_children.append(child)
                            print(f'time_step_1_children {time_step_1_children}')
                            node.children.remove(child)

                    else:
                        # continue bfs
                        if child not in seen:
                            seen.add(child)
                            queue.append(child)

                if len(time_step_1_children) > 0 and isinstance(node, Product) \
                        and node.scope == scope_top_interface_parent:

                    # interface root node is the time step 1 root node
                    interface_node = self.make_interface_node(time_step_1_children)
                    interface_nodes.append(interface_node)

                    # attach latent interface node to time step 0 interface parent
                    latent_interface_child = LatentInterface(interface_idx=interface_idx)
                    node.children.append(latent_interface_child)
                    interface_idx += 1

        print(f'interface_nodes {interface_nodes}')
        interface_layer = self.make_interface_layer(interface_nodes)

        assign_ids(root)
        return root, interface_layer

    @staticmethod
    def make_interface_node(children_list):

        if len(children_list) > 1:
            # more than one child of parent prod node belong to time step 1
            prod_node = Product(children=children_list)
            interface_node = prod_node  # copy.deepcopy(prod_node)
            assign_ids(interface_node)
        else:
            print(f'time_step_1_children {children_list}')
            interface_node = children_list[0]  # copy.deepcopy(time_step_1_children)
            assign_ids(interface_node)

        return interface_node

    def make_interface_layer(self, interface_nodes):

        interface_layer = InterfaceSwitch(children=interface_nodes)
        interface_layer = self.attach_interface_nodes_to_interface_layer(interface_layer)
        assign_ids(interface_layer)

        return interface_layer

    @staticmethod
    def attach_interface_nodes_to_interface_layer(interface_root):

        # perform bfs

        seen, queue = set([interface_root]), collections.deque([interface_root])
        latent_interface_list = []
        while queue:
            node = queue.popleft()
            interface_idx = 0

            if not isinstance(node, Leaf):

                for child in node.children:
                    if isinstance(node, InterfaceSwitch):
                        # create a latent interface node for each interface node
                        latent_interface_node = LatentInterface(interface_idx=interface_idx)
                        latent_interface_list.append(latent_interface_node)
                        interface_idx += 1

                    if child not in seen:
                        seen.add(child)
                        queue.append(child)

                if all(isinstance(child, Leaf) for child in node.children):

                    initial_interface_weights = [0.5] * len(latent_interface_list)
                    interface_sum = Sum(children=latent_interface_list, weights=initial_interface_weights)

                    if isinstance(node, Product):
                        node.children.append(interface_sum)

                    else:
                        node.children.append(interface_sum)
                        children = copy.deepcopy(node.children)
                        prod_interface = Product(children=children)
                        node.children.clear()
                        node.children.append(prod_interface)


        assign_ids(interface_root)
        return interface_root

    def learn_rspmn(self, data):

        spmn = SPMN(self.two_time_step_params.partial_order_two_time_steps,
                    self.two_time_step_params.decision_nodes_two_time_steps,
                    self.two_time_step_params.utility_nodes_two_time_steps,
                    self.two_time_step_params.feature_names_two_time_steps, cluster_by_curr_information_set=True,
                    util_to_bin=False)
        self.spmn_structure_two_time_steps = spmn.learn_spmn(data)

        return self.spmn_structure_two_time_steps





class RSPMNTwoTimeStepParams:
    def __init__(self, rspmn_params):
        self.partial_order_two_time_steps = get_partial_order_two_time_steps(rspmn_params.partial_order)
        self.decision_nodes_two_time_steps = get_nodes_two_time_steps(rspmn_params.decision_nodes)
        self.utility_nodes_two_time_steps = get_nodes_two_time_steps(rspmn_params.utility_nodes)
        self.feature_names_two_time_steps = get_feature_names_two_time_steps(self.partial_order_two_time_steps)
