import numpy as np

from spn.algorithms.MPE import get_node_funtions
from spn.structure.Base import Product, Sum, get_nodes_by_type, eval_spn_top_down, Leaf
from spn.structure.leaves.parametric.Parametric import Parametric, Gaussian, In_Latent
from spn.algorithms.oSLRAUStat import update_mean_and_covariance, update_curr_mean_and_covariance

from spn.algorithms.MPE import mpe_sum, mpe_prod
from spn.structure.Base import get_topological_order_layers

from spn.algorithms.oSLRAU import oSLRAU_sum


def initialise_mean_and_covariance(node, parent_result, data):
    scope = node.scope.copy()
    l = data.shape[1]
    for scpe in node.scope:
        if scpe >= l:
            scope.remove(scpe)

    x = data[np.ix_(parent_result, scope)]
    m = x.shape[0]
    n = node.count

    assert not np.isnan(x).any(), "data cointains NaN values"

    if node.cov is None:
        # print(node.scope)
        node.cov = np.identity(len(scope))
    if node.mean is None:
        node.mean = np.zeros(len(scope))

    mean = node.mean
    cov = node.cov

    # update mean
    curr_sample_sum = x.sum(axis=0)
    new_mean = ((n) * (mean) + curr_sample_sum) / (n + m)

    # update covariance
    dx = x - new_mean
    #dm = new_mean - mean

    new_cov = (n * cov + dx.T.dot(dx)) / (n + m) #- np.outer(dm, dm)

    # update node values
    node.mean = new_mean
    node.cov = new_cov

node_functions = get_node_funtions()
_node_top_down_oSLRAU = node_functions[0].copy()
_node_top_down_oSLRAU.update({Sum: oSLRAU_sum, Product: mpe_prod})


def eval_rspn_top_down_partial_update(root, oSLRAU_params, update_leaves, eval_functions = _node_top_down_oSLRAU, all_results=None, parent_result=None, **args):
    """
    evaluates an spn top to down


    :param root: spnt root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, parent_results, args**) and returns [node, intermediate_result]. This intermediate_result will be passed to node as parent_result. If intermediate_result is None, no further propagation occurs
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
    :param parent_result: initial input to the root node
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    all_results[root] = [parent_result]
    # nodes_to_update = []
    in_latent_dict = {}

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]
            # print("Leaf", isinstance(n, Leaf))
            # print("patametric", isinstance(n, Parametric))
            # print("Gaussian", isinstance(n, Gaussian))
            if isinstance(n, Product):
                result = func(n, param, **args)

            elif isinstance(n, Leaf):
                # print("inLeaf", isinstance(n, Leaf))
                instances = np.concatenate(param)
                if len(instances) > 0:
                    n.count = n.count + len(instances)
                    if type(n) == In_Latent:
                        in_latent_dict[n] = instances
                    elif update_leaves == True:
                        update_mean_and_covariance(n, instances, oSLRAU_params, **args)  # works only for gaussian nodes

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

    return all_results, in_latent_dict
