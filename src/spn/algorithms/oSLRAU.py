from spn.algorithms.oSLRAUStruct import update_structure
from spn.algorithms.Inference import log_likelihood, sum_likelihood, prod_likelihood
from spn.algorithms.MPE import get_node_funtions
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, get_nodes_by_type, Max, Leaf, assign_ids
import numpy as np
import collections
from spn.algorithms.oSLRAUStat import update_mean_and_covariance, update_curr_mean_and_covariance

from spn.structure.Base import get_topological_order_layers


class oSLRAUParams:
    """
	Parameters
	----------
	batchsize : number of samples in a mini-batch.
	            if 0, use the entire set as one batch.
	mergebatch : number of samples a product node needs to see before updating
	             its structure.
	corrthresh : correlation coefficient threshold above which two variables
	             are considered correlated.
	mvmaxscope : number of variables that can be combined into a multivariate
	             leaf node.

	"""

    def __init__(self, batchsize=128, mergebatch_threshold=128, corrthresh=0.1, mvmaxscope=1, equalweight = False, currVals = False):

        self.batchsize = batchsize
        self.mergebatch_threshold = mergebatch_threshold
        self.corrthresh = corrthresh
        self.mvmaxscope = mvmaxscope
        self.equalweight = equalweight
        self.currVals = currVals


def merge_input_vals(l):
    return np.concatenate(l)


def oSLRAU_prod(node, parent_result, oSLRAU_params, data=None, lls_per_node=None):


    nodes_to_update = None
    if parent_result is None:
        return None, nodes_to_update

    parent_result = merge_input_vals(parent_result)

    # print(node, node.count)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result

    if len(parent_result) != 0:
        update_mean_and_covariance(node, parent_result, oSLRAU_params, data)
        node.count = node.count + len(parent_result)
        merge_threshold = oSLRAU_params.mergebatch_threshold

        if node.count > merge_threshold:
            nodes_to_update = node
            update_curr_mean_and_covariance(node, parent_result, oSLRAU_params,
                                            data)  # update curr_mean and curr_cov parameters of prodcut node using current data

    return children_row_ids, nodes_to_update



def oSLRAU_sum(node, parent_result, data=None, lls_per_node=None):

    if parent_result is None :
        return None

    parent_result = merge_input_vals(parent_result)

    w_children_log_probs = np.zeros((len(parent_result), len(node.weights)))
    for i, c in enumerate(node.children):
        w_children_log_probs[:, i] = lls_per_node[parent_result, c.id] + np.log(node.weights[i])

    max_child_branches = np.argmax(w_children_log_probs, axis=1)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result[max_child_branches == i]

    if len(parent_result) != 0:
        node.count = node.count + len(parent_result)
        weights = []
        for i, c in enumerate(node.children):
            # print(len(children_row_ids[i])/len(parent_result))
            # w[i] = len(children_row_ids[i])/len(parent_result)
            # print(node.weights)
            weights.append(node.weights[i] + (len(children_row_ids[c])/len(parent_result)))
            #print(node.weights[i])

        for i, c in enumerate(node.children):
            #print(node.weights)
            node.weights[i] = float(weights[i]) / np.sum(weights)

    return children_row_ids


    # if len(parent_result) == 0:
    #     return None
    #
    #
    # # print(node, node.count)
    #
    # w_children_log_probs = np.zeros((len(parent_result), len(node.weights)))
    # for i, c in enumerate(node.children):
    #     w_children_log_probs[:, i] = lls_per_node[parent_result, c.id] + np.log(node.weights[i])
    #
    # max_child_branches = np.argmax(w_children_log_probs, axis=1)
    #
    # children_row_ids = []
    #
    # for i, c in enumerate(node.children):
    #     children_row_ids.append(parent_result[max_child_branches == i])
    #
    #
    #
    #
    #
    # # print(children_row_ids)
    # return children_row_ids



node_functions = get_node_funtions()
_node_top_down_oSLRAU = node_functions[0].copy()
_node_bottom_up_mpe = node_functions[1].copy()
_node_top_down_oSLRAU.update({Sum: oSLRAU_sum, Product: oSLRAU_prod})
_node_bottom_up_mpe.update({Sum: sum_likelihood, Product: prod_likelihood})



def oSLRAU(node, input_data, node_top_down_mpe=_node_top_down_oSLRAU, node_bottom_up_mpe=_node_bottom_up_mpe,
           in_place=False):
    valid, err = is_valid(node)
    assert valid, err

    if in_place:
        data = input_data
    else:
        data = np.array(input_data)


    #print('oSLRAU')

    # _node_top_down_oSLRAU[Product] = oSLRAU_prod
    # _node_top_down_oSLRAU[Sum] = oSLRAU_sum

    oSLRAU_params = oSLRAUParams()
    nodes = get_nodes_by_type(node)
    tot_nodes_len = len(nodes)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))

    # one pass bottom up evaluating the likelihoods
    log_likelihood(node, data, dtype=data.dtype, node_log_likelihood=node_bottom_up_mpe, lls_matrix=lls_per_node)

    instance_ids = np.arange(data.shape[0])

    # one pass top down to decide on the max branch until it reaches a leaf, then it fills the nan slot with the mode
    all_results, nodes_to_update = oSLRAU_eval_spn_top_down(node, node_top_down_mpe, oSLRAU_params, parent_result=instance_ids,
                                                            data=data, lls_per_node=lls_per_node)

    spn = oSLRAU_update_structure(node, oSLRAU_params, tot_nodes_len, nodes_to_update)

    return spn


def oSLRAU_update_structure(root,  oSLRAU_params, tot_nodes_len, nodes_to_update):
    for node in nodes_to_update:
        update_structure(node, oSLRAU_params, tot_nodes_len)

    assign_ids(root)
    v, err = is_valid(root)
    assert v, err
    return root


def oSLRAU_eval_spn_top_down(root, eval_functions, oSLRAU_params, all_results=None, parent_result=None, **args):
    """
    evaluates an spn top to down


    :param root: spnt root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, parent_results, args**) and returns [node, intermediate_result]. This intermediate_result will be passed to node as parent_result. If intermediate_result is None, no further propagation occurs
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
    :param parent_result: initial input to the root node
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """
    # if all_results is None:
    #     all_results = {}
    # else:
    #     all_results.clear()
    #
    # nodes_to_update = []
    #
    # queue = collections.deque([(root, parent_result)])
    #
    # while queue:
    #     node, parent_result = queue.popleft()
    #
    #     if isinstance(node, Product):
    #         result, update_node = eval_functions[type(node)](node, parent_result, params, **args)
    #         if update_node is not None:
    #             nodes_to_update.append(update_node)
    #
    #     elif isinstance(node, Leaf):
    #         #result = eval_functions[type(node)](node, parent_result, **args)
    #         node.count = node.count + len(parent_result)
    #         update_mean_and_covariance(node, parent_result, params, **args)  # works only for gaussian nodes
    #
    #     else:
    #         result = eval_functions[type(node)](node, parent_result, **args)
    #
    #     all_results[node] = result
    #
    #     if result is not None and not isinstance(node, Leaf):
    #
    #         assert len(result) == len(node.children), "invalid function result for node %s" % (node.id)
    #         for i, node in enumerate(node.children):
    #             queue.append((node, result[i]))
    #
    # return all_results[root], nodes_to_update

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    all_decisions = []
    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    all_results[root] = [parent_result]
    nodes_to_update = []

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]

            if isinstance(n, Product):
                result, update_node = func(n, param, oSLRAU_params, **args)
                if update_node is not None:
                    nodes_to_update.append(update_node)

            elif isinstance(n, Leaf):
            # result = func(n, param, **args)
                n.count = n.count + len(parent_result)
                update_mean_and_covariance(n, parent_result, oSLRAU_params, **args)  # works only for gaussian nodes

            else:
                result = func(n, param, **args)


            if result is not None and not isinstance(n, Leaf):
                assert isinstance(result, dict)

                for child, param in result.items():
                    if child not in all_results:
                        all_results[child] = []
                    all_results[child].append(param)

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results, nodes_to_update

