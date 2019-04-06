import numpy as np
from spn.structure.Base import Product, Leaf
from spn.structure.leaves.parametric.Parametric import Gaussian, Multivariate_Gaussian


def update_mean_and_covariance(node, parent_result, params, data, lls_per_node= None):


    scope = node.scope
    x = data[np.ix_(parent_result, scope)]
    m = x.shape[0]
    n = node.count

    if isinstance(node, Product):

        if node.cov is None:
            # print(node.scope)
            node.cov = np.identity(len(node.scope))
        if node.mean is None:
            node.mean = np.zeros(len(node.scope))

        mean = node.mean
        cov = node.cov

        # update mean
        curr_sample_sum = x.sum(axis=0)
        new_mean = ((n) * (mean) + curr_sample_sum) / (n + m)

        # update covariance
        dx = x - mean
        dm = new_mean - mean

        new_cov = (n * cov + dx.T.dot(dx)) / (n + m) - np.outer(dm, dm)

        # update node values
        node.mean = new_mean
        node.cov = new_cov

    #for gaussian leaves
    if isinstance(node, Leaf):

        if type(node) == Gaussian:
            if node.stdev is None:
                #print("in leaf update", node.scope)
                node.stdev = 1
            if node.mean is None:
                node.mean = 0

            mean = node.mean
            stdev = node.stdev
            cov = np.array(np.square(stdev))

        elif type(node) == Multivariate_Gaussian:
            if node.cov is None:
                #print("in leaf update", node.scope)
                node.cov = np.identity(len(node.scope))
            if node.mean is None:
                node.mean = np.zeros(len(node.scope))

            mean = node.mean
            cov = node.cov

        #print("in leaf update", node)
        curr_sample_sum = x.sum(axis=0)
        new_mean = ((n) * (mean) + curr_sample_sum) / (n + m)

        # update covariance
        dx = x - mean
        dm = new_mean - mean
        new_cov = (n * cov + dx.T.dot(dx)) / (n + m) - np.outer(dm, dm)

        # update node values
        if type(node) == Gaussian:
            new_stdev = np.sqrt(np.abs(new_cov))
            node.mean = new_mean[0]
            node.stdev = new_stdev[0][0]
        else:
            node.mean = new_mean
            node.cov = new_cov

    return


def iterate_corrs(node, corrthresh):
        v = np.diag(node.cov).copy()
        v[v<1e-4] = 1e-4
        corrs = np.abs(node.cov) / np.sqrt(np.outer(v, v))
        rows, cols = np.unravel_index(np.argsort(corrs.flatten()), corrs.shape)

        for i, j in zip(reversed(rows), reversed(cols)):
            if corrs[i, j] < corrthresh:
                break
            yield i,j


def update_curr_mean_and_covariance(node, parent_result, params, data, lls_per_node= None):


    scope = node.scope
    x = data[np.ix_(parent_result, scope)]
    m = x.shape[0]
    n = node.count

    if isinstance(node, Product):

        if node.cov is None:
            # print(node.scope)
            node.curr_cov = np.identity(len(node.scope))
        if node.mean is None:
            node.curr_mean = np.zeros(len(node.scope))

        # update mean
        mean = np.mean(x, axis=0)
        if m == 1:
            # x1 = np.repeat(x, 2, axis =0)
            cov = np.identity(len(node.scope))
        else:
            cov = np.cov(x, rowvar=False)

        node.curr_mean = mean
        node.curr_cov = cov