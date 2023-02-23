import numpy as np
from sklearn.cluster import KMeans


def gen_block_matrix(blocksize=50):
    diag = 0.7
    offdiag = 0.3
    repeat = 3
    d = blocksize * repeat
    B = np.array(
        [
            [diag if i == j else offdiag for j in range(blocksize)]
            for i in range(blocksize)
        ]
    )
    X = np.tile(B, (repeat, repeat))
    B = np.random.rand(d, d)
    A = np.array([[1 if X[i, j] >= B[i, j] else 0 for j in range(d)] for i in range(d)])

    rng = np.random.default_rng()
    P = rng.permutation(np.identity(d))
    return np.matmul(P, np.matmul(A, P))


def cluster():
    X = gen_block_matrix(5)
    print(X)


if __name__ == "__main__":
    cluster()
