import numpy as np
from enum import Enum
from mathtools import trinum

#########################################
### Spin Space Methods
#########################################


class Spinmode(Enum):
    NONE = 0
    INT = 1
    SPIN = 2

    @staticmethod
    def checkmode(spin: tuple):
        """Returns the Spinmode of the provided spin"""

        # default mode
        mode = Spinmode.NONE

        # if in splitmode, check type of first element
        test = spin[0] if isinstance(spin, tuple) else spin

        # case that spin is int
        if isinstance(test, int):
            mode = Spinmode.INT
        # case that test is spin
        elif isinstance(test, np.ndarray):
            mode = Spinmode.SPIN
        else:
            raise Exception(f"Unrecognized spin format of type {type(spin)}")

        return mode


def int2spin(val, shape: tuple):
    """Generate spin representation of integer. Two different behaviors depending on whether val and shape are tuples or integers.

    Parameters
    ----------
    val : tuple(int) or int
        a tuple of integers representing a spin
    shape : tuple(int) or int
        the shape of the spin to produce. Must match length of val if a tuple

    Returns
    -------
    numpy.ndarray
        a 1-d array consisting of -1 and 1 representing a spin
    """

    # helper function
    def singleint2spin(num, N):
        """Convert single integer to spin"""
        b = list(np.binary_repr(num).zfill(N))  # get binary representation of num
        a = [-1 if int(x) == 0 else 1 for x in b]  # convert to spin representation
        return np.array(a).astype(np.int8)  # return as a numpy array

    if len(shape) == 1:
        return singleint2spin(val, shape[0])

    # val and shape both tuples if reach this point
    if len(val) != len(shape):
        raise Exception(f"ERROR: Spin {val} does not match specified shape {shape}!")

    # generate tuple of spins
    spin = tuple(singleint2spin(val[i], N) for i, N in enumerate(shape))
    if len(spin) == 1:
        return spin[0]

    return spin


def spin2int(spin: tuple):
    """Generate integer representation of a spin

    Parameters
    ----------
    spin : numpy.ndarry or tuple of numpy.ndarray

    Returns
    -------
    int or tuple of int

    """

    def singlespin2int(spin):
        # store the length of spin
        N = len(spin)

        # number to return
        num = tuple([2 ** (N - (i + 1)) * (1 if spin[i] == 1 else 0) for i in range(N)])
        return sum(num)

    # this spin is in split format
    if isinstance(spin, tuple):
        return tuple([singlespin2int(s) for s in spin])
    # spin is not split
    else:
        return singlespin2int(spin)


class Spinspace:
    """
    Wrapper class for a spinspace of a specified size

    Attributes
    ----------
    shape : tuple
        the shape of the spinspace
    dim : int
        the dimension of the spinspace
    vdim : int
        the dimension of the virtual spinspace corresponding to this spinspace
    size : int
        the cardinality of the spinspace
    shape : int
        the shape taken by spins. (2,2) means S^2 x S^2, (4,) means S^4
    indices : list
        shape converted to index form, for use in numpy.split
    split : bool
        convenience flag, False if spinspace comprised of one component, True if decomposed into multiple components

    Methods
    -------
    __init__(shape : tuple)
        initializes spinspace. If not decomposing spinspace then set shape = (dim) where dim is the desired dimension
    __iter__()
        makes this class an iterator.
    __next__()
        fetches next element. Converts _current_index from an integer to a spin in the necessary format
    int2spin(val)
        wrapper for static function of same name, converts spin from integer format to spin format
    spin2int(spin)
        wrapper for static function of same name, converts spin from spin format to integer format
    convspin(spin, mode)
        returns the provided spin converted into the specified mode
    splitspin(spin)
        splits the provided spin into the desired shape, e.g. (1,1,1,1,1) --> ((1,1),(1,1),(1)) split into shape (2,2,1). Reads shape from self.shape
    dist(spin1,spin2)
        returns the hamming distance between the two spins


    """

    def __init__(self, shape: tuple, mode=Spinmode.SPIN):
        self.shape = shape
        self.dim = sum(shape)
        self.vdim = self.dim + trinum(self.dim)
        self.size = 2**self.dim
        self.indices = [sum(shape[:i]) for i in range(1, len(shape))]
        self.mode = mode
        self.split = False if len(shape) == 1 else True
        self._current_index = 0

    def __iter__(self):
        """Makes this object iterable"""
        return self

    def __next__(self):
        """Returns the next spin in the iteration formatted in the appropriate mode. Converts _current_index to the appropriate spin."""
        if self._current_index >= self.size:
            raise StopIteration

        # if in split mode this converts _current_index to a tuple of integers
        formatted_index = (
            self._current_index
            if self.split == False
            else self.splitspin(self._current_index)
        )
        self._current_index += 1

        # return the spin in the desired format
        return self.convspin(spin=formatted_index, mode=self.mode)

    def int2spin(self, val: tuple):
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

        return int2spin(val=val, shape=self.shape)

    def spin2int(self, spin):
        """Generate integer representation of a spin

        Parameters
        ----------
        spin : numpy.ndarry or tuple of numpy.ndarray

        Returns
        -------
        int or tuple of int

        """

        return spin2int(spin=spin)

    def convspin(self, spin, mode=Spinmode.NONE):
        """Convert a spin of unspecified mode into the mode specified"""
        # by default convert spin to the mode of this Spinspace
        if mode == Spinmode.NONE:
            mode = self.mode

        # store the mode of the provided spin
        spinmode = Spinmode.checkmode(spin)

        # spin already matches provided mode
        if spinmode == mode:
            return spin

        # if desired mode is INT convert to int
        if mode == Spinmode.INT:
            print(f"in convspin {spin2int(spin)}")
            return spin2int(spin)
        # if desired mode is SPIN convert to spin
        elif mode == Spinmode.SPIN:
            return int2spin(spin, self.shape)
        else:
            raise Exception(f"Unrecognized spin mode: {mode}")

    def splitspin(self, spin):
        """Convert spin format to split form.

        Performs its own checks for spin mode, doesn't rely on Spinspace settings.
        This makes it flexible and is used in __next__ for instance. It means that
        _current_index can be stored as a single integer and can be converted into
        any format necessary.
        """
        # if already split, do nothing
        if isinstance(spin, tuple):
            return spin

        # if spin is in SPIN mode
        if isinstance(spin, np.ndarray):
            return tuple(np.split(spin, self.indices))

        # if spin is in INT mode
        elif isinstance(spin, int):
            # convert int to spin, split spin, convert back to integer
            # uses "static" int2spin method to avoid checking shape
            tempspin = np.split(int2spin(spin, (self.dim,)), self.indices)
            newspin = tuple(self.spin2int(s) for s in tempspin)

            # print(f"{int2spin(spin, self.dim)}")
            # print(f"{np.split(int2spin(spin, self.dim), self.shape)}")
            # print(f"Spin in splitspin : {spin}")
            # print(f"Dim in splitspin used for shape : {self.dim}")
            # print(f"Shape in splitspin : {self.shape}")
            # print(f"tempspin { tempspin }")
            return newspin

    def catspin(self, spin):
        """Turns a split spin into a...nonsplit spin. Concatonates the given spin."""
        # if spin not split, return
        if isinstance(spin, tuple) == False:
            return spin

        # store the mode
        mode = Spinmode.checkmode(spin)

        # convert to spin type, easier to concatonate
        if mode == Spinmode.INT:
            spin = self.convspin(spin, mode=Spinmode.SPIN)

        return self.convspin(np.concatenate(spin), mode=mode)

    def rand(self):
        """Returns a random spin from this spinspace, sampled uniformly"""
        from random import randint

        a = randint(0, self.size - 1)
        spin = self.splitspin(a) if self.split else a
        return self.convspin(spin=spin, mode=self.mode)

    def pairspin(self, spin, mode=Spinmode.NONE):
        """Returns the spin corresponding to the pairwise interactions of spin in {self.mode} format. Always in nonsplit mode."""
        # match spinmode of Spinspace by default
        if mode == Spinmode.NONE:
            mode = self.mode
        a = self.catspin(self.convspin(spin, Spinmode.SPIN))
        N = len(a)
        pair = []
        for i in range(N):
            for j in range(i + 1, N):
                pair.append(a[i] * a[j])
        return self.convspin(np.array(pair), mode=mode)

    def vspin(self, spin, split=False):
        """Returns the virtual spin corresponding to the given spin"""
        # guarentee spin is concatenated spin mode
        spin = self.catspin(self.convspin(spin, Spinmode.SPIN))
        spin2 = self.pairspin(spin, mode=Spinmode.SPIN)

        if split:
            return (spin, spin2)

        return np.concatenate((spin, spin2))

    def dist(self, spin1, spin2):
        """Return the hamming distance between two spins. Easiest to first convert to spin mode"""
        if Spinmode.checkmode(spin1) == Spinmode.INT:
            spin1 = self.int2spin(spin1)

        if Spinmode.checkmode(spin2) == Spinmode.INT:
            spin2 = self.int2spin(spin2)

        # ensure spins are concatonated for ease
        spin1 = self.catspin(spin1)
        spin2 = self.catspin(spin2)

        return sum(np.not_equal(spin1, spin2))

    def dist2(self, spin1, spin2):
        """Return the 2nd order hamming distance between two spins. Not efficiently implemented."""
        if Spinmode.checkmode(spin1) == Spinmode.INT:
            spin1 = self.int2spin(spin1)

        if Spinmode.checkmode(spin2) == Spinmode.INT:
            spin2 = self.int2spin(spin2)

        # ensure spins are concatonated for ease
        spin1 = self.catspin(spin1)
        spin2 = self.catspin(spin2)

        if len(spin1) != len(spin2):
            raise Exception(
                f"Error: len(spin1) : {len(spin1)} is not equal to len(spin2) : {len(spin2)}"
            )

        result = 0
        for i in range(len(spin1)):
            for j in range(i + 1, len(spin1)):
                result += bool(spin1[i] * spin1[j] - spin2[i] * spin2[j])

        return result

    def vdist(self, spin1, spin2):
        """Returns the distance between spin1 and spin2 in virtual space"""
        return self.dist(spin1, spin2) + self.dist2(spin1, spin2)

    def multiply(self, spin1, spin2):
        """Multiplies spin1 and spin2 pointwise"""
        spin1 = self.catspin(self.convspin(spin1, mode=Spinmode.SPIN))
        spin2 = self.catspin(self.convspin(spin2, mode=Spinmode.SPIN))
        result = np.multiply(spin1, spin2)
        if self.split:
            result = self.splitspin(result)

        return self.convspin(result, mode=self.mode)

    def inv(self, spin):
        """Invert the spins of the given spin"""
        inverse = -1 * self.catspin(self.convspin(spin, mode=Spinmode.SPIN))
        if self.split:
            inverse = self.splitspin(inverse)

        return self.convspin(inverse, mode=self.mode)


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
