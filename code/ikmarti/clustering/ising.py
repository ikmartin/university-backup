from spinspace import Spinspace, Spinmode
import numpy as np
import math
from mathtools import ditri_array, trinum


# data structure for vector in p-space
class vspin:
    """Data structure for a vector in virtual space"""

    def __init__(self, h, J, spin=[]):
        if isinstance(h, np.ndarray):
            h = h.tolist()
        if isinstance(J, np.ndarray):
            J = J.tolist()

        print(h)
        print(J)
        self.gsize = len(h)
        self.vec = ditri_array(h, J)

    # converts index from upper triangular matrix to vspin
    def ijtok(self, i, j):
        if j < i:
            raise ValueError("Received (i,j) = {}! Cannot have j < i.".format((i, j)))

        if i >= self.gsize:
            raise ValueError(
                "Invalid index! First index {} larger than {}".format(i, self.gsize)
            )
        if j >= self.gsize:
            raise ValueError(
                "Invalid index! The second index {} is larger than {}".format(
                    j, self.gsize
                )
            )
        return trinum(self.gsize) - trinum(self.gsize - i) + (j - i)

    # converts index from vspin to (i,j)
    def ktoij(self, k):
        if k >= trinum(self.gsize):
            raise ValueError("Index out of range!")

        # inverting ijtok
        # just do the math
        triN = trinum(self.gsize)
        t = triN - k
        # this is the m such that trinum(m) = trinum(N - i)
        m = (math.isqrt(8 * t) + 1) // 2
        i = self.gsize - m
        j = trinum(m) - t + i

        return (i, j)

    # overload for indexing
    def __getitem__(self, index):
        # if index is a single integer
        if isinstance(index, int):
            return self.vec[index]

        # otherwise it is not and must be a tuple
        # ensure index is a tuple
        if not isinstance(index, tuple):
            raise TypeError("Expected integer or tuple, received {}".format(index))

        # ensure either one or two indicies are passed
        if len(index) not in [1, 2]:
            raise TypeError(
                "Expected tuple of length 1 or 2, received {}".format(index)
            )

        return self.vec[self.ijtok(index[0], index[1])]

    def __mul__(self, other):
        return sum([self.vec[i] * other[i] for i in range(self.vec)])

    def validate(self, a, b):
        print("-------------")
        print(" Validation")
        print("-------------")
        vis = input("Visualize? (y/n)") in ["Y", "y"]

        for k in range(len(self.vec)):
            if self.vec[k] != self[self.ktoij(k)]:
                print("ERROR! Index k = {}".format(k))

        for i in range(self.gsize):
            for j in range(i, self.gsize):
                if self.vec[self.ijtok(i, j)] != self[i, j]:
                    print("ERROR! Index i,j = {}".format((i, j)))

        if vis:
            upper = vspin.ditri_to_numpy(a, b)
            test = self.tomatrix()
            print("EXPECTED:")
            print(upper)
            print("\nRECEIVED:")
            print(test)
            print("\n-----------------------\n")

    def toarray(self):
        return self.vec

    def tonumpy(self):
        return np.array(self.vec)

    def tomatrix(self):
        # easier notation for size
        N = self.gsize

        # index convenience function
        ind = lambda N, i, j: trinum(N) - trinum(N - i) + j

        # the numpy array
        test = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                test[i, j] = self.vec[ind(N, i, j - i)]

        return test

    @staticmethod
    def spinmin(spin):
        size = len(spin)
        h = -1 * np.array(spin)
        J = -1 * np.array(
            [-1 * spin[i] * spin[j] for i in range(size) for j in range(i + 1, size)]
        )
        return vspin(h=h, J=J)

    @staticmethod
    def pspin(spin):
        size = len(spin)
        h = np.array(spin)
        J = np.array(
            [spin[i] * spin[j] for i in range(size) for j in range(i + 1, size)]
        )
        return vspin(h=h, J=J)

    @staticmethod
    def ditri_to_numpy(di, tri):
        N = len(di)
        if len(tri) != trinum(N - 1):
            raise ValueError("Length error! Length of tri not compatible with di")

        l = lambda N, i: trinum(N) - trinum(N - i) - i
        d = lambda N, i: (N - 1) - i

        upper = np.zeros((N, N))
        ind = lambda N, i, j: trinum(N) - trinum(N - i) + j
        for i in range(N):
            x = l(N, i)
            y = x + d(N, i)
            upper[i, i:] = [di[i]] + tri[x:y]

        return upper


class PICircuit:
    """Base class for Pre Ising Circuits"""

    def __init__(self, N: int, M: int, A: int = 0):
        self.N = N
        self.M = M
        self.A = A
        self.inspace = Spinspace(tuple([self.N]))
        self.outspace = Spinspace(tuple([self.M + self.A]), mode=Spinmode.SPIN)
        self.spinspace = Spinspace((self.N, self.M, self.A))

    def energy(self, spin, ham):
        return None

    def f(self, spin):
        raise Exception("No logic implemented for the circuit!")


class IMul(PICircuit):
    """Ising Multiply Circuit"""

    def __init__(self, N1: int, N2: int, A: int = 0):
        super().__init__(N=N1 + N1, M=N1 + N2, A=A)
        self.inspace = Spinspace(shape=(N1, N2), mode=Spinmode.SPIN)
        self.N1 = N1
        self.N2 = N2

    def f(self, spin):
        # get the numbers corresponding to the input spin
        num1, num2 = self.inspace.convspin(spin=spin, mode=Spinmode.INT)

        # multiply spins as integers and convert into spin format
        result = self.outspace.convspin(num1 * num2)
        return result

    def generate_graph(self):
        graph = []
        for s in self.inspace:
            graph.append([s, self.f(s)])
            print(graph[-1])
        return graph


def example():
    G = IMul(2, 2, 0)
    G.generate_graph()

    spin1 = G.inspace.rand()
    spin2 = G.inspace.rand()
    print(f"spin1 : {spin1}")
    print(f"spin1 : {spin2}")
    print(f"ham dist : {G.inspace.dist(spin1,spin2)}")
    print(f"pairspin : {G.inspace.pairspin(spin1)}")
    print(f"pairspin : {G.inspace.pairspin(spin2)}")
    print(f"ham dist2 : {G.inspace.dist2(spin1,spin2)}")
    print(f"vspin1 : {G.inspace.vspin(spin1,split=True)}")
    print(f"vspin1 : {G.inspace.vspin(spin2,split=True)}")
    print(f"vdist : {G.inspace.vdist(spin1,spin2)}")
    print(f"pointwise multiply spin1 * spin2 : {G.inspace.multiply(spin1,spin2)}")
    print(f"invert spin1 : {G.inspace.inv(spin1)}")


if __name__ == "__main__":
    example()
