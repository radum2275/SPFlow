from spn.algorithms.oSLRAUStat import iterate_corrs
import numpy as np
from spn.structure.Base import Product, Sum, assign_ids, Leaf
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian, Multivariate_Gaussian


def update_structure(node, params, tot_nodes_len=None):
    corr_threshold = params.corrthresh
    for i, j in iterate_corrs(node, corr_threshold):
        if i == j:
            continue

        ci = None
        cj = None
        for child in node.children:
            if (i + node.scope[0]) in child.scope:
                ci = child
            if (j + node.scope[0]) in child.scope:
                cj = child

        if ci is not None and cj is not None and ci != cj:
            merge_children(node, ci, cj, params, tot_nodes_len)
            break

    return


def merge_children(node, ci, cj, params, tot_nodes_len=None):
    mvmaxscope = params.mvmaxscope

    ci = ci
    cj = cj
    scope = np.concatenate((ci.scope, cj.scope))
    scope.sort()

    if len(scope) <= mvmaxscope:
        merge_into_mvleaf(node, ci, cj, scope, params, tot_nodes_len)
    else:
        merge_into_sumnode(node, ci, cj, scope, params, tot_nodes_len)

    return


def merge_into_mvleaf(node, ci, cj, scope, params, tot_nodes_len=None):
    if isinstance(ci, Leaf) and isinstance(cj, Leaf):
        # mean = node.mean
        # mean = mean[scope[0]:scope[-1] + 1]
        # cov = node.cov
        # cov = cov[scope[0]:(scope[-1] + 1), scope[0]:(scope[-1] + 1)]

        mean = node.mean
        mean_dict = dict(zip(node.scope, mean))
        mean = np.array([mean_dict[s] for s in scope])

        cov = node.cov
        r = range(cov.shape[1])
        ran_dict = dict(zip(node.scope, r))
        r = [ran_dict[s] for s in scope]
        cov = cov[np.ix_(r, r)]
        # stdev = np.sqrt(np.abs(cov))

        c = Multivariate_Gaussian(mean, cov)
        c.scope = scope
        c.id = tot_nodes_len

        node.children.extend([c])
        node.children.remove(ci)
        node.children.remove(cj)

    return


def merge_into_sumnode(node, ci, cj, scope, params, tot_nodes_len=None):
    assert tot_nodes_len is not None, "total nodes in initial structure cannot be None"

    mean = node.mean
    mean_dict = dict(zip(node.scope, mean))
    mean = np.array([mean_dict[s] for s in scope])

    cov = node.cov
    r = range(cov.shape[1])
    ran_dict = dict(zip(node.scope, r))
    r = [ran_dict[s] for s in scope]
    cov = cov[np.ix_(r, r)]

    # cov = [cov[s] for s in scope]
    # cov = cov[scope[0]:(scope[-1]+1), scope[0]:(scope[-1]+1)]

    p1 = Product(children=[ci, cj], mean=mean, cov=cov)
    p1.scope = scope

    p2 = create_factored_dist(scope, tot_nodes_len, node, params)

    if params.equalweight:
        p1.count = 1
        p2.count = 1
    else:
        p1.count = node.count
        p2.count = 1

    s = Sum(children=[p1, p2])
    s.count = node.count

    w1 = float((p1.count + 1) / (s.count + 2))
    w2 = float((p2.count + 1) / (s.count + 2))
    tot = w1 + w2

    w1 = w1 / tot
    w2 = w2 / tot
    s.weights = [w1, w2]
    s.scope = scope

    s.id = tot_nodes_len
    p1.id = tot_nodes_len + 1
    p2.id = tot_nodes_len + 2

    node.children.extend([s])
    node.children.remove(ci)
    node.children.remove(cj)

    return


def create_factored_dist(scope, tot_nodes_len, node, params):
    children = []

    if params.currVals:
        mean = node.curr_mean.copy()
        mean_dict = dict(zip(node.scope, mean))
        mean = np.array([mean_dict[s] for s in scope])

        cov = node.curr_cov.copy()
        r = range(cov.shape[1])
        ran_dict = dict(zip(node.scope, r))
        r = [ran_dict[s] for s in scope]
        cov = cov[np.ix_(r, r)]

        stdev = np.sqrt(np.abs(cov))

    else:
        mean_p2 = np.zeros(len(scope))
        cov_p2 = np.identity(len(scope))

    for i, s in enumerate(scope, 0):
        # print(s)
        if params.currVals:
            assert stdev[i][i] or mean[i] is not np.nan, "nan stdev or mean at %s" % (node)
            c = Gaussian(mean=mean[i], stdev=stdev[i][i], scope=[s])
        else:
            c = Gaussian(mean=0, stdev=1, scope=[s])

        c.id = tot_nodes_len + 2 + i + 1
        children.append(c)
    # print(children)

    if params.currVals:
        p2 = Product(children=children, mean=mean, cov=cov)
    else:
        p2 = Product(children=children, mean=mean_p2, cov=cov_p2)
    # p2.count = 0
    p2.scope = scope

    return p2
