import numpy as np
import itertools  # for Cartesian product and nothing else
import random

#########################################
### Spin Space Methods
#########################################


class Spinspace:
    """
    Wrapper class for a spinspace of a specified size

    Attributes
    ----------
    shape : tuple
        the shape of the spinspace
    dim : int
        the dimension of the spinspace

    Methods
    -------
    __init__(shape : tuple)
        initializes spinspace. If not decomposing spinspace then set shape = (dim) where dim is the desired dimension
    level(val : int, axis : int = 0)
        not yet implemented
    """

    def __init__(self, shape: tuple, mode: str = "array"):
        self.shape = shape
        self.dim = sum(shape)
        self.mode = mode
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_index >= self.dim:
            raise StopIteration

        if self.mode == "array":
            member = self.dec2spin(tuple([self._current_index]), split=False)
            print(member)
        else:
            member = self._current_index

        self._current_index += 1
        return member

    def level(self, val, axis=(0)):
        pass

    def dec2spin(self, val: tuple, split=False):
        """Generate spin representation of integer

        Parameters
        ----------
        val : tuple(int)
            a tuple of integers representing a spin
        split : bool = False
            split the spin into tuple of spins according to shape or not

        Returns
        -------
        numpy.ndarray
            a 1-d array consisting of -1 and 1 representing a spin
        """

        # check that shape matches
        if len(val) != len(self.shape):
            raise Exception("Provided spin does not match shape of spinspace")

        # helper function
        def singledec2spin(num, N):
            """Convert single integer to spin"""
            b = list(np.binary_repr(num).zfill(N))  # get binary representation of num
            a = [-1 if int(x) == 0 else 1 for x in b]  # convert to spin representation
            return np.array(a).astype(np.int8)  # return as a numpy array

        # generate tuple of spins
        spin = tuple(singledec2spin(val[i], N) for i, N in enumerate(self.shape))
        print(spin)
        if split:
            return spin
        else:
            return np.concatenate(spin)

    def spin2dec(self, spin: tuple):
        """Generate integer representation of a spin

        Parameters
        ----------
        spin : tuple

        Returns
        -------
        numpy.ndarray
            a 1-d array consisting of -1 and 1 representing a spin
        """

        def singlespin2dec(spin):
            # store the length of spin
            N = len(spin)

            # number to return
            num = (2 ** (N - (i + 1)) * (1 if spin[i] == 1 else 0) for i in range(N))
            return sum(num)

        return (singlespin2dec(s) for s in spin)

    def split(self, spin):
        """Wrapper for numpy.split function"""
        return np.split(spin, self.shape)

    def cat(self, spin):
        """Wrapper for the numpy.concatenate function"""
        if isinstance(spin, np.ndarray):
            return spin
        elif isinstance(spin[0], np.ndarray):
            return np.concatenate(spin)
        elif isinstance(spin[0], int):
            return self.dec2spin(val=spin, split=False)
        else:
            raise Exception(f"Unrecognized spin format: {type(spin)}")

    def dist(self, spin1, spin2):
        """Return the hamming distance between two spins. Expects spin1 and spin2 to be"""
        if isinstance(spin1, np.ndarray) == False:
            spin1 = self.cat(spin1)

        if isinstance(spin2, np.ndarray) == False:
            spin2 = self.cat(spin2)

        return sum(np.not_equal(spin1, spin2))


spin_spaces = {}


def multiply(s, t):
    """Multiplies two spins of equal length

    Returns: (numpy.ndarray) numpy 1D array of length equal to length of inputs. Entries are 1's and -1's.

    Params:
    *** s: (numpy.ndarray) the first spin
    *** t: (numpy.ndarray) the second spin
    """

    if s.size != t.size:
        raise ValueError("Lengths of spins don't match, cannot multiply!")

    return np.multiply(s, t)


def inv(s):
    """Returns the multiplicative inverse of the provided spin.

    Returns: (numpy.ndarray)

    Params:
    *** s: (numpy.ndarray) the spin to invert
    """
    return np.array([-1 * si for si in s])


def rand_spin(N):
    """Generates a random spin of the given size

    Returns: (numpy.ndarray) 1D numpy array of length N

    Params:
    *** N: (int) the length of the array to generate
    """

    a = []
    for i in range(N):
        a.append(random.choice([-1, 1]))

    return np.array(a)
