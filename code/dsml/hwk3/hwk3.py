from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import itertools as it


def gen_block_matrix(blocksize=50, permute=True):
    diag = 0.7
    offdiag = 0.3
    repeat = 3
    d = repeat*blocksize
    A = 0.7*np.ones((blocksize, blocksize))
    B = 0.3*np.ones((blocksize, blocksize))
    X = np.block(
        [[A,B,B],
         [B,A,B],
         [B,B,A]]
    )

    # permute
    if permute:
        rng = np.random.default_rng()
        P = rng.permutation(np.identity(d))
        X = np.matmul(P, np.matmul(X, np.transpose(P)))

    return X


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

#Dummy data
def train_kmeans(X):
    kmeans = KMeans(n_clusters=5, verbose=2, n_init=1) #<-- init=1, verbose=2
    kmeans.fit(X)
    return kmeans

#HELPER FUNCTION
#Takes the returned and printed output of a function and returns it as variables
#In this case, the returned output is the model and printed is the verbose intertia at each iteration

def redirect_wrapper(f, inp):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    returned = f(inp)                #<- Call function
    printed = new_stdout.getvalue()  #<- store printed output

    sys.stdout = old_stdout
    return returned, printed


returned, printed = redirect_wrapper(train_kmeans, X)

#Extract inertia values
inertia = [float(i[i.find('inertia')+len('inertia')+1:]) for i in printed.split('\n')[1:-2]]

#Plot!
plt.plot(inertia)

def cluster(k=3):
    # the parameters for KMeans
    # just extra stuff to include for convenience 
    common_params = {"init": "k-means++", "n_init" : "auto"}

    # generate the matrix
    X = gen_block_matrix(5)

    # cluster the damn thing
    kmeans = KMeans(n_clusters=k, **common_params).
    y_pred = kmeans.fit_predict(X)
#Extract inertia values
inertia = [float(i[i.find('inertia')+len('inertia')+1:]) for i in printed.split('\n')[1:-2]]

    print(y_pred)
    print(X)


if __name__ == "__main__":
    print(gen_block_matrix(blocksize=3, permute=False))
    cluster()
