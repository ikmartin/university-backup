import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
from matplotlib import image


def problem1a():
    #####################
    ## analyze first case
    #####################
    mu = 0
    sigma = 1
    d = 1000
    cd = 100
    samples = 1000
    A = np.array([np.random.normal(mu, sigma, d) for i in range(samples)])
    q, r = lin.qr(A)
    epsilon = 10 ** (-15)
    savefig = ""
    nbins = 40
    width = 0.9

    # check for orthogonality
    for col in q.T:
        for acol in q.T:
            if np.array_equal(col, acol):
                break

            val = np.dot(col, acol)
            if val > epsilon:
                print("Error! Got {}".format(val))

    # truncate and normalize the vectors
    newq = q[:cd, :]
    for i in range(newq.shape[1]):
        norm = np.linalg.norm(newq[:, i])
        print("Normalizing vector {}".format(i))
        newq[:, i] = newq[:, i] / norm
        print("   norm = {}".format(np.linalg.norm(newq[:, i])))

    # get pairwise dot products, remove diagonal elements, flatten
    dots = np.matmul(newq.T, newq)
    dots = dots[~np.eye(dots.shape[0], dtype=bool)].reshape(dots.shape[0], -1)
    dots = dots.flatten()

    print(
        "Length newq: {}    length dots: {}    shape: {}".format(
            len(newq), len(dots), newq.shape
        )
    )
    plt.hist(dots, nbins, rwidth=width)
    plt.title(
        "Dot products of {} orthonormal vectors \nin {}-dimensions projected to a random\n{}-dimensional space".format(
            samples, d, cd
        )
    )
    plt.subplots_adjust(top=0.8)

    ####################
    # save figure prompt
    if savefig == "":
        savefig = input("Save figure? (y/n)")

    if savefig in ["Y", "y"]:
        plt.savefig("prob1_method1_{}dim_{}vecs".format(d, samples))
        print("saved figure")

    plt.clf()
    plt.close()


def problem1b():
    mu = 0
    sigma = 1
    d = 100
    samples = 1000
    epsilon = 10 ** (-15)
    savefig = ""
    nbins = 40
    width = 0.9

    # create and normalize data
    A = np.array([np.random.normal(mu, sigma, d) for i in range(samples)]).T
    count = 0
    for i in range(A.shape[1]):
        count += 1
        print("normalizing vector {}".format(count))
        A[:, i] = A[:, i] / np.linalg.norm(A[:, i])

    # get pairwise dot products, remove diagonal elements, flatten
    dots = np.matmul(A.T, A)
    dots = dots[~np.eye(dots.shape[0], dtype=bool)].reshape(dots.shape[0], -1)
    dots = dots.flatten()

    # make histogram
    plt.hist(dots, nbins, rwidth=width)
    plt.title(
        "Pairwise dot products of {} standard Gaussian vectors \nin {}-dimensions".format(
            samples, d
        )
    )
    plt.subplots_adjust(top=0.8)

    ####################
    # save figure prompt
    if savefig == "":
        savefig = input("Save figure? (y/n)")

    if savefig in ["Y", "y"]:
        plt.savefig("prob1_method2_{}dim_{}vecs".format(d, samples))
        print("saved figure")

    plt.clf()
    plt.close()


def problem6a():
    # parameters
    maxpower = 4

    A = np.array(
        [list(range(i, 11)) + [0 for j in range(i - 1)] for i in range(1, 11)],
        dtype=np.float64,
    )
    U, S, Vh = np.linalg.svd(A)
    X = np.matmul(A, A.T)
    for i in range(maxpower):
        X = np.matmul(X, X)

    u1 = X[:, 0] / np.linalg.norm(X[:, 0])
    if u1[0] / U[:, 0][0] < 0:
        u1 = -1 * u1

    print("---------------\nPROBLEM 6 (a)\n---------------")
    print("Estimate of u1 by B^{}:".format(2**maxpower))
    print(" ", u1)
    print("\nActual u1 given by numpy:")
    print(" ", U[:, 0])

    diff = u1 - U[:, 0]
    print("\nError: {}".format(np.linalg.norm(diff)))
    print("------------------------\n")


def problem6b():
    print("PROBLEM 6 (b)\n-----------------------")
    #  parameters
    A = np.array(
        [list(range(i, 11)) + [0 for j in range(i - 1)] for i in range(1, 11)],
        dtype=np.float64,
    )
    mu = 0
    sigma = 1
    d = 10
    samples = 4
    power = 16
    X = np.array(
        [np.random.normal(mu, sigma, d) for i in range(samples)], dtype=np.float64
    ).T

    # compute SVD
    U, S, Vh = np.linalg.svd(A)

    # find ortho basis for X and compute B
    q, r = lin.qr(X)
    B = np.matmul(A, A.T)

    # take the vector products
    # reorthonormalize each time to prevent collapse onto a one dimensional subspace
    for i in range(power):
        q = np.matmul(B, q)
        qnew, _ = lin.qr(q)
        q = qnew

    for i in range(q.shape[1]):
        # make sign consistent with U
        if q[:, i][0] / U[:, i][0] < 0:
            q[:, i] = -1 * q[:, i]
        print("Left singular vector {} is".format(i + 1))
        print(q[:, i])
        print("Error: {}".format(np.linalg.norm(q[:, i] - U[:, i])))


# returns compressed matrix from first k singular vectors
def svd_compress(U, s, Vh, k):
    frob_ratio = np.linalg.norm(s[:k]) / np.linalg.norm(s)
    return np.matmul(U[:, :k] * s[:k], Vh[:k, :]), frob_ratio


def image_to_array(path):
    return image.imread(path)


def image_from_array(img, filename="", bw=True):
    if bw:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

    plt.savefig(filename)
    print("Saved image as {}".format(filename))


def problem7ab(impath=""):
    # parameters
    impath = "images/bwrock{}.jpg" if impath == "" else impath
    bw = True
    compressions = [1, 4, 16, 32, 128]

    # get the image and perform SVD
    img = image_to_array(impath.format(""))
    print("Image shape: {}".format(img.shape))
    U, S, Vh = np.linalg.svd(img)

    # add full svd to compression list
    compressions.append(len(S))

    # save resulting image for each compression size
    ratios = []
    for i in range(len(compressions)):
        A, r = svd_compress(U, S, Vh, compressions[i])
        ratios.append(r)
        image_from_array(A, filename=impath.format(compressions[i]))
        print("  captured {}% of Frobenius norm".format(r * 100))

    print(ratios)


def problem7c():
    size = 600
    title = "images/noise{}.png"
    compressions = [1, 4, 16, 32, 128]

    # generate the image and save the original
    img = np.random.rand(size, size) * 255
    image_from_array(img, filename=title)

    # perform SVD
    U, S, Vh = np.linalg.svd(img)

    # add full svd to compression list
    compressions.append(len(S))

    # save the resulting image for each compression size
    ratios = []
    for i in range(len(compressions)):
        A, r = svd_compress(U, S, Vh, compressions[i])
        ratios.append(r)
        image_from_array(A, filename=title.format(compressions[i]))
        print("  captured {}% of Frobenius norm".format(r * 100))


if __name__ == "__main__":
    # problem7ab("images/bwrock{}.jpg")
    # problem7ab("images/bwwaterfall{}.jpg")
    # problem7c()
    problem6a()
    problem6b()
