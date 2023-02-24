from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def gen_block_matrix(blocksize=50, permute=True):
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

    # permute
    if permute:
        rng = np.random.default_rng()
        P = rng.permutation(np.identity(d))
        A = np.matmul(P, np.matmul(A, P))

    return A


# cluster points based on magnitude
# doesn't give good clustering
def cluster_mag(k=3, bsize=5):
    X = gen_block_matrix(blocksize=5, permute=False)
    x = np.array([[np.linalg.norm(X[:, i]), 0] for i in range(np.shape(X)[1])])
    y_pred = KMeans(n_clusters=k).fit_predict(x)

    plt.scatter(x[:, 0], x[:, 1])
    plt.show()


# gets the optimal kmeans clustering for k many clusters
# horribly inefficient
def get_optimal_clustering(k=3):
    print("NO CODE")


def cluster(k=3):
    # the parameters for KMeans
    common_params = {"init": "k-means++"}

    X = gen_block_matrix(10)
    count = {sum(X[:, i]): sum(X[:, i]) for i in range(len(X))}
    for val in count.values():
        print(val)
    y_pred = KMeans(n_clusters=k, **common_params).fit_predict(X)

    print(y_pred)
    print(X)


if __name__ == "__main__":
    cluster()
