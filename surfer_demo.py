import pandas as pd
import qgrid
qgrid.nbinstall(overwrite=True)

def fix_dangling_nodes(A):
    """
    Return input matrix A (square) with all diagonal elements set to 1.
    """
    return tril(A, -1) + triu(A, 1) + eye(size(A, 0))

def normalize_rows(G):
    """
    Return input matrix A normalized so each row
    represents probability of leaving node
    (each row of returned matrix sums to one)
    """
    out_degree = sum(G, 1)
    H = copy(G)
    for i in range(len(out_degree)):
        if out_degree[i] > 0:
            H[i,:] = G[i,:]/out_degree[i]
    return H

def remove_zeros(H, alpha):
    return alpha*H + (1 - alpha)*ones(shape(H))/size(H,0)

def power_iteration(Ha, steps):
    """
    Beginning with random starting vector, perform specified
    number of power iteration steps. Return final vector.
    """
    u = rand(size(Ha, 1))
    u = u/norm(u)

    for idx in range(steps):
        u = dot(Ha, u)
        u = u/norm(u)
    return u
