import numpy as np



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