from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# generates the desired block matrix
def gen_block_matrix(blocksize=50, permute=True):
    diag = 0.7
    offdiag = 0.3
    repeat = 3

    # should be 150
    d = repeat * blocksize

    # create the blocks
    A = 0.7 * np.ones((blocksize, blocksize))
    B = 0.3 * np.ones((blocksize, blocksize))

    # create the block matrix
    C = np.block([[A, B, B], [B, A, B], [B, B, A]])

    # convert to the matrix of ones using Bernoulli distribution
    X = np.random.binomial(n=1, p=C)

    # permute
    rng = np.random.default_rng()
    P = rng.permutation(np.identity(d))
    if permute:
        X = np.matmul(P, np.matmul(X, np.transpose(P)))

    # labels for the "natural" cluster of the rows prior to permutation, permuted to match.
    # ASSUMES CLUSTERING WITH k = 3
    nat_labels = [0] * blocksize + [1] * blocksize + [2] * blocksize
    return X, P, nat_labels


# do the clustering
# X: data
# k: number of clusters
# n_init: number of iterations to run
def train_kmeans(X, k=3, n_init=10):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10)  # <-- init=1, verbose=2
    kmeans.fit(X)
    return kmeans


# code for problem 1a
def prob1a():
    # generate data
    X, P, nat_labels = gen_block_matrix(blocksize=50, permute=True)

    # perform k-means
    kmeans = train_kmeans(X, n_init=10)
    unshuffled_labels = np.matmul(np.array(kmeans.labels_), P)

    # Read off the labels of the cluster and the cost.
    # unshuffled_labels ought to look like the natural clustering,
    # i.e. [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2].
    print("\n\n------------------------------------\nProblem 1 (a)\n--------------")
    print("\nUNSHUFFLED LABELS:\n", unshuffled_labels)
    print("\nCost achieved:", kmeans.inertia_)
    print("\n---end---\n------------------------------------")


# def prob1b():
def prob1b():
    # generate data
    X, P, nat_labels = gen_block_matrix(blocksize=50, permute=True)

    # cluster for various values of k and get costs
    ks = list(range(1, 20))
    kmeans = [train_kmeans(X, k=i, n_init=10) for i in ks]
    costs = [km.inertia_ for km in kmeans]

    print("\n\n------------------------------------\nProblem 1 (b)\n--------------")
    print("\n     <displaying graph>\n")
    print("\n---end---\n------------------------------------")

    # should look like an elbow. The "point" of the elbow is qualitatively a good choice of k
    # see the "elbow method"
    plt.plot(ks, costs)
    plt.xlabel("# of clusters")
    plt.ylabel("cost of clustering")
    plt.title("Cost of clustering vs number of clusters")
    plt.show()


def prob1c():
    mu, sigma = 0, 1
    col = 150
    row = 10

    # generate spherical projection matrix of size row x col
    # the gaussian vectors are of length 150, not of length 10. This may be wrong.
    Phi = np.array([np.random.normal(mu, sigma, col) for i in range(row)])
    for i in range(row):
        Phi[i, :] = Phi[i, :] / np.linalg.norm(Phi[i, :])

    # generate the data to cluster
    X, P, nat_labels = gen_block_matrix(blocksize=50, permute=True)
    Xn = np.matmul(X, np.transpose(Phi))

    # do clustering
    kmeans = train_kmeans(Xn, n_init=10)
    shuffled_labels = np.array(kmeans.labels_)
    unshuffled_labels = np.matmul(shuffled_labels, P)

    # Read off the labels of the cluster and the cost.
    print("\n\n------------------------------------\nProblem 1 (c)\n--------------")
    print("\nSHUFFLED LABELS:\n", shuffled_labels)
    print("\nUNSHUFFLED LABELS:\n", unshuffled_labels)
    print("\nCost achieved:", kmeans.inertia_)
    print("\n---end---\n------------------------------------")


def prob1d():
    # generate the data to cluster
    X, P, nat_labels = gen_block_matrix(blocksize=50, permute=True)

    # get the svd stuff
    U, S, Vh = np.linalg.svd(X)

    # truncate Vh
    # try adding more columns -- you'll quickly recover the original clustering.
    V = Vh[:, :3]
    Xn = np.matmul(X, V)

    # do clustering
    kmeans = train_kmeans(Xn, n_init=10)
    shuffled_labels = np.array(kmeans.labels_)
    unshuffled_labels = np.matmul(shuffled_labels, P)

    # Read off the labels of the cluster and the cost.
    print("\n\n------------------------------------\nProblem 1 (d)\n--------------")
    print("\nSHUFFLED LABELS:\n", shuffled_labels)
    print("\nUNSHUFFLED LABELS:\n", unshuffled_labels)
    print("\nCost achieved:", kmeans.inertia_)
    print("\n---end---\n------------------------------------")


if __name__ == "__main__":
    prob1a()
    prob1b()
    prob1c()
    prob1d()
