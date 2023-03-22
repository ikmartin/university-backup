import numpy as np
import itertools  # for Cartesian product and nothing else
import random
from enum import Enum

#########################################
### Spin Space Methods
#########################################

class Spinmode(Enum):
    NONE = 0
    INT = 1
    SPIN = 2

def int2spin(val, shape):
    """Generate spin representation of integer. Two different behaviors depending on whether val and shape are tuples or integers.

    Parameters
    ----------
    val : tuple(int)
        a tuple of integers representing a spin
    shape : tuple(int)
        the shape of the spin to produce

    Returns
    -------
    numpy.ndarray
        a 1-d array consisting of -1 and 1 representing a spin
    """

    if type(shape) != type(val):
        raise Exception(f"Error: val {val} and shape {shape} must be of the same type!")
    # helper function
    def singleint2spin(num, N):
        """Convert single integer to spin"""
        b = list(np.binary_repr(num).zfill(N))  # get binary representation of num
        a = [-1 if int(x) == 0 else 1 for x in b]  # convert to spin representation
        return np.array(a).astype(np.int8)  # return as a numpy array

    if isinstance(shape, int):
        return singleint2spin(val, shape)

    if len(val) != len(shape):
        raise Exception(f"ERROR: Spin {val} does not match specified shape {shape}!")

    # generate tuple of spins
    spin = tuple(singleint2spin(val[i], N) for i, N in enumerate(shape))
    print(spin)
    if split:
        return spin
    else:
        return np.concatenate(spin)

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
        num = (2 ** (N - (i + 1)) * (1 if spin[i] == 1 else 0) for i in range(N))
        return sum(num)

    # this spin is in split format
    if isinstance(spin, tuple):
        return (singlespin2int(s) for s in spin)
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

    Methods
    -------
    __init__(shape : tuple)
        initializes spinspace. If not decomposing spinspace then set shape = (dim) where dim is the desired dimension
    """
    
    def __init__(self, shape: tuple, mode = Spinmode.SPIN, split:bool = False):
        self.shape = shape
        self.dim = sum(shape)
        self.mode = mode
        self.split = split
        self._current_index = 0

    def __iter__(self):
        """Makes this object iterable"""
        return self

    def __next__(self):
        """Returns the next spin in the iteration formatted in the appropriate mode"""
        if self._current_index >= self.dim:
            self._current_index = 0
            raise StopIteration

        # if in split mode this converts _current_index to a tuple of integers
        formatted_index = self._current_index if self.split == False else self.splitspin(self._current_index)
        print(formatted_index)
        self._current_index += 1
        return self.convspin(spin=formatted_index)

    def int2spin(self, val: tuple, split=False):
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

    def spin2int(self, spin: tuple):
        """Generate integer representation of a spin

        Parameters
        ----------
        spin : numpy.ndarry or tuple of numpy.ndarray 

        Returns
        -------
        int or tuple of int
            
        """

        return spin2int(spin=spin)

    def checkmode(self, spin):
        """Returns the Spinmode of the provided spin"""
        mode = Spinmode.NONE

        # if in splitmode, check type of first element
        test = spin[0] if isinstance(spin,tuple) else spin

        # case that spin is int
        if isinstance(test, int):
            mode = Spinmode.INT

        # case that test is spin
        elif isinstance(test, np.ndarray):
            mode = Spinmode.SPIN
        else:
            raise Exception(f"Unrecognized spin format of type {type(spin)}")

    def convspin(self, spin, mode=Spinmode.NONE):
        """Convert a spin of unspecified mode into the mode specified"""
        # by default convert spin to the mode of this Spinspace
        if mode == Spinmode.NONE:
            mode = self.mode

        # store the mode of the provided spin
        spinmode = self.checkmode(spin)

        # spin already matches provided mode
        if spinmode == mode:
            return spin

        # if desired mode is INT convert to int
        if mode == Spinmode.INT:
            return spin2int(spin)
        # if desired mode is SPIN convert to spin
        elif mode == Spinmode.SPIN:
            return int2spin(spin)
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
            return tuple(np.split(spin, self.shape))

        # if spin is in INT mode
        elif isinstance(spin, int):
            # convert int to spin, split spin, convert back to integer
            # uses "static" int2spin method to avoid checking shape
            tempspin = np.split(int2spin(spin, self.dim), self.shape)
            newspin = tuple( self.spin2int(s) for s in tempspin)

            print(f"{int2spin(spin, self.dim)}")
            print(f"{np.split(int2spin(spin, self.dim), self.shape)}")
            print(f"Spin in splitspin : {spin}")
            print(f"Dim in splitspin used for shape : {self.dim}")
            print(f"Shape in splitspin : {self.shape}")
            print(f"tempspin { tempspin }")
            return newspin

    def dist(self, spin1, spin2):
        """Return the hamming distance between two spins. Easiest to first convert to spin mode"""
        if checkmode(spin1) ==  Spinmode.INT:
            spin1 = self.int2spin(spin1)

        if checkmode(spin2) ==  Spinmode.INT:
            spin2 = self.int2spin(spin2)

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
