import numpy as np
import matplotlib.pyplot as plt
import math
import random as rand


def prob3(d=1000, samples=250, vis=False):
    mu, sigma = 0, 1
    figsize = (8, 4)

    # set figure dimensions
    fig = plt.figure()
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    # histogram settings
    nbins = 100
    width = 0.8

    # generate and normalize the data
    X = np.array([np.random.normal(mu, sigma, d) for i in range(samples)])
    for i in range(samples):
        norm = np.linalg.norm(X[i, :])
        for j in range(d):
            X[i, j] = X[i, j] / norm

    print("Varience: {}".format(np.var(X[:, 0])))
    ####################
    # save figure prompt

    plt.show()
    plt.clf()
    plt.close()


def prob4():
    t = np.arange(1, 15.0 + 0.01, 0.01)
    s = np.array([1 / a for a in t])

    plt.rcParams.update(
        {
            "font.size": 16,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsfonts}",
        }
    )
    plt.plot(t, s)
    plt.xlabel(r"$a$")
    plt.ylabel(r"$\mathbb{P}[X\geq a]$", fontsize=16)
    plt.title(
        r"Probability that $X \geq a$",
        fontsize=16,
        color="gray",
    )
    # Make room for the ridiculously large title.
    plt.subplots_adjust(top=0.8)
    plt.savefig("prob4")
    plt.show()


# d = dimension of ambient vector space
# vis = boolean, attempt to visualize
def prob6(d=1000, samples=250, vis=False, savefig="", alldist=True):
    mu, sigma = 0, 1
    figsize = (8, 4)

    # set figure dimensions
    fig = plt.figure()
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    # histogram settings
    nbins = 100
    width = 0.8

    # generate and normalize the data
    X = np.array([np.random.normal(mu, sigma, d) for i in range(samples)])
    for i in range(samples):
        norm = np.linalg.norm(X[i, :])
        for j in range(d):
            X[i, j] = X[i, j] / norm

    ##################################
    # plot the main figure of interest
    if alldist:
        dist = np.array([])
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    dist = np.append(dist, np.linalg.norm(X[i, :] - X[j, :]))

        counts, bins = np.histogram(dist)
        plt.hist(dist, nbins, rwidth=width)
        plt.title(
            "Distances between uniformly distributed points on \n{}-dimensional sphere w/\n{} samples".format(
                d - 1, samples
            )
        )
        plt.subplots_adjust(top=0.8)

        ####################
        # save figure prompt
        if savefig == "":
            savefig = input("Save figure? (y/n)")

        if savefig in ["Y", "y"]:
            plt.savefig("prob6_dim{}_{}samples.png".format(d - 1, samples))
            print("saved figure")

        plt.clf()
        plt.close()

    #######################################
    # plot distributions for a single point
    indices = list(np.random.permutation(np.arange(0, len(X)))[:5])
    for i in indices:
        idist = np.array([])
        for j in range(len(X)):
            if i != j:
                idist = np.append(idist, np.linalg.norm(X[i, :] - X[j, :]))

        plt.hist(idist, nbins, rwidth=width)
        plt.title(
            "Distances from {}th point from dataset uniformly distributed on \n{}-dimensional sphere w/\n{} samples".format(
                i, d - 1, samples
            )
        )
        plt.subplots_adjust(top=0.8)
        plt.savefig("prob6_dim{}_pt{}.png".format(d - 1, i))
        plt.clf()
        plt.close()

    if vis:
        # exit if there is too much data to meaningfully visualize
        if d > 3 or len(X) > 500 * d / 3:
            return 0

        fig = plt.figure(figsize=(8, 8))
        col = np.array([np.linalg.norm(X[0] - X[i]) for i in range(0, len(X))])
        if d == 2:
            ax = fig.add_subplot(111)
            ax.scatter(X[:, 0], X[:, 1], c=col)
            ax.scatter(X[0, 0], X[0, 1], c="black")
        elif d == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=col)
            ax.scatter(X[0, 0], X[0, 1], X[0, 2], c="black")
        plt.show()


if __name__ == "__main__":
    # prob4()
    d = 3
    prob6(d=d, samples=500, vis=True, savefig="n", alldist=True)
    # prob3(d=d, samples=500)
