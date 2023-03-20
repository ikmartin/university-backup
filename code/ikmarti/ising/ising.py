import numpy as np # for life and liberty and matrices
import networkx as nx # for graphs, probably can delete honestly
import matplotlib.pyplot as plt # for pretty pictures 
import pickle # for serialization
import itertools # for Cartesian product and nothing else
import spinspace # import spinspace module

#########################################
### Helper Methods
#########################################

# make a strict upper triangular matrix
# assumes A is square
def strict_triu(A):
    """ Removes the diagonal and lower triangular portion of a square matrix

    Returns: n x n strictly upper triangular matrix

    Params:
    *** A: (numpy.ndarray) the n x n matrix to be chopped up
    """

    if A.shape[0] != A.shape[1]:
        raise Exception("strict_triu requires matrix to be square")
        
    return np.triu(A) - np.diag(np.diag(A))

# check if system of equations is consistent
def system_consistent(A):
    ''' Checks whether or not A constitutes a consistent set of equations
    Returns: (bool) True if given matrix is a system set of equations, False if not
    
    Params:
    *** A: (numpy.ndarray) the matrix to check
    ''' 
    
    return np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A[:,:-1])
   
def is_numpy(a):
    ''' Check if input is a numpy array
    Returns: (bool) True if a is numpy array, False if not

    Params:
    *** a: (object) the object to check
    '''
    
    return type(a).__module__ == np.__name__

def sign(a,letter=False):
    ''' Wrapper to determine the sign of a number. Used to write out systems of equations.
    
    Returns: (int or string) 
    *** 1 or '+' if a > 0
    *** -1 or '-' if a < 0
    *** 0 or '' otherwise

    Params:
    *** a: (float) the number to check
    '''

    if a > 0:
        return "+" if letter else 1
    elif a < 0:
        return "-" if letter else -1
    else:
        return "" if letter else 0

####################################################################
#### COUNT METHODS
####################################################################

def num_params(G):
    ''' Return the number of h's and J's for the given graph size

    Returns: (int) equal to G + (G-1)G/2

    Params:
    *** G: (int) the number of vertices in the graph
    '''
    
    return int(G + (G-1)*G/2)

### Spin <--> Binary <--> Decimal Methods ### 

def spin_to_binary(spin):
    ''' Converts a spin in +/- 1 format to binary 1/0 format
    Returns: (array(str)) array of strings, each entry either 1 or 0

    Params:
    *** spin: (iterable) an array in spin format
    '''

    return ['0' if x == -1 else str(x) for x in spin]

####################################################################

def mult_to_binary(a: int, b: int,length = -1):
    ''' Multiplies two unsigned integers and returns result as a binary number. Cannot handle negative numbers.

    Returns: 
    '''
    num = [int(x) for x in list(bin(a * b))[2:]]
    if length > len(num):
        n = len(num)
        for i in range(length - n):
            num.insert(0,0)
    return num

####################################################################
### Ising parameter methods, i.e. methods handling h and J
####################################################################

def J2mat(G,tempJ):
    ''' Make a J array into a matrix
    '''
    if is_numpy(tempJ):
        tempJ = tempJ.tolist()

    J = []
    a = 0
    d = G-1
    # stick tempJ into a strict upper diagonal matrix buffered with 0's
    while d >= 0:
        J.append([0 for i in range(G - d)] + tempJ[a:a+d])
        a += d
        d = d - 1

    return np.array(J)

def split_param_array(A,matformat=True):
    ''' Receives an array of h and J values A = [h,J] and splits it.
    Returns: tuple h,J

    Params:
    *** A: (array) the total array of parameters to split
    *** matformat (default=True): (bool) decides whether or not to return J as a matrix or a 1D array.
    '''
    
    # used for sqrt
    import math

    # deduce the size of the underlying graph from len(A) = N + (N-1)(N-2)/2
    G = -0.5 + 0.5*(math.sqrt(8*len(A) +1))

    if G.is_integer() == False:
        print(f"ERROR: invalid length for A! Got G = {G}.")
        return None
    
    else:
        # make G an integer
        G = int(G)

        # get h and J values
        h = A[:G]
        tempJ = A[G:]
        
        # decide whether to convert J to matrix form or not
        if matformat == False:
            return (np.array(h),np.array(tempJ))
        
        else:
            J = J2mat(G,tempJ)
            return (np.array(h), np.array(J))

####################################################################

def get_unit_param(num: (int), G,matformat = True):
    ''' "Unit Param" here means an h and J are +/-1 at every component. This function interprets this pair as an integer in binary and returns the next pair.

    Returns: (array, array) corresponding to (h, J).

    Params:
    *** num: (int) a decimal integer used to fetch h,J
    *** G: (int) size of the graph in question, equal to length of h array
    
    Optional Params:
    *** mat_format (default = True): (bool) if True, J returned as a GxG matrix
    '''

    # use dec2spin method in spinspace
    import spinspace as sp

    m = num_params(G)
    
    # get array of 1's and -1's, use to retreive h and J
    A = sp.dec2spin(num=num,N=m)
    
    return split_param_array(A,matformat = matformat)

####################################################################

def get_param_mod3(num: (int), G,mat_format = True):
    ''' "mod3" here means an h and J are +/-1 or 0 at every component. This function interprets such a pair as an integer and converts it to an array in ternary, with 0 = 0, 1 = 1, 2 = -1 (think Z/3Z).

    Returns: (array, array) corresponding to (h, J).

    Params:
    *** num: (int) a decimal integer used to fetch h,J
    *** G: (int) size of the graph in question, equal to length of h array
    
    Optional Params:
    *** mat_format (default = True): (bool) if True, J returned as a GxG matrix
    '''

    # use dec2spin method in spinspace
    import spinspace as sp

    m = num_params(G)
    
    # get array of 1's and -1's, use to retreive h and J
    A = sp.dec2mod3(num=num,N=m)
    
    h = A[:G]
    tempJ = A[G:]

    if mat_format == False:
        return (num+1,h,tempJ)
    
    else:
        tempJ = tempJ.tolist()
        J = []
        a = 0
        d = G-1
        # stick tempJ into a strict upper diagonal matrix buffered with 0's
        while d >= 0:
            J.append([0 for i in range(G - d)] + tempJ[a:a+d])
            a += d
            d = d - 1

        return (h, np.array(J))

####################################################################

def rand_param(G,num0,num1,numJ0=None,numJ1=None,):
    ''' Return a random mod 3 h and J with certain number of occurrences of -1, 0, 1 (the number of -1's can be deduced from the number of 0's and 1's). If numJ0 and numJ1 are set, then num0 and num1 are treated as the number of 0's and 1's respectively for h.

    Returns: mod3 h and J parameters

    Params:
    *** G: (int) the size of the graph
    *** num0: (int) the number of occurrences of 0
    *** num1: (int) the number of occurrences of 1

    Optional Params:
    *** numJ0 (default=None): (int) the number of 0's in J
    *** numJ1 (default-None): (int) the number of 1's in J
    '''

    import random
    
    # wraps the list.pop function to remove multiple indices
    def dels(a,inds):
        print("length a: ",len(a))
        print("values in inds: ", inds)
        if type(a) != list: a = list(a)
        for i in sorted(inds,reverse=True): 
            del a[i]
        return a

    # makes separate treatment of h and J less cumbersome
    def wrap_rand(N,n0,n1):
        # list of indices
        ind = range(N)

        # randomize the 0 entries, remove from available entries
        ind0 = random.sample(ind,n0)
        dels(ind,ind0)

        # randomize the 1 entries, remove from available entries
        ind1 = random.sample(ind,n1)
        dels(ind,ind1)

        # initialize a as list of -1's then update 0 and 1 entries
        a = [-1 for i in range(N)]
        for i in ind0: a[i] = 0
        for i in ind1: a[i] = 1
        return a

    # get the number of parameters for this graph
    numP = num_params(G)

    # treat h and J separately
    if numJ0 != None and numJ1 != None:
        h = wrap_rand(G,num0,num1)
        J = wrap_rand(numP - G,n0 = numJ0,n1 = numJ1)
        A = h + J
    else:
        A = wrap_rand(N = numP, n0 = num0, n1 = num1)
        h,J = split_param_array(A,matformat = False)

    return np.array(h),J2mat(G=G,tempJ=J),spinspace.mod32dec(G=G,A=A)
    
####################################################################
#### Wrapper for spin space module
####################################################################

def get_spin_space(N: int, spin_val = [-1,1]):
    ''' Returns spin space for a graph of size N.

    Params:
    *** N: (int) the size of the spin space

    Optional Params:
    *** spin_val (default = [-1,1]): (array) the set of possible spin values.
    '''
    return spinspace.get_spin_space(N)

def multiply(s,t):
    ''' Multiplies two spins of equal length
    
    Returns: (numpy.ndarray) numpy 1D array of length equal to length of inputs. Entries are 1's and -1's.

    Params:
    *** s: (numpy.ndarray) the first spin
    *** t: (numpy.ndarray) the second spin
    '''
    
    return spinspace.multiply(s,t)

def inv(s):
    ''' Returns the multiplicative inverse of the provided spin.

    Returns: (numpy.ndarray)
    
    Params:
    *** s: (numpy.ndarray) the spin to invert
    '''
    return spinspace.inv(s)

#########################################
### Class Definitions
#########################################

class IGraph: 
    ''' Ising Graph class
    '''

    def __init__(self,*,h,J,spin_val=[-1,1],create_spin_space=True,name: str ="",debug=False,**kwargs):
        ''' Initialize the Ising Graph
        
        Params:
        *** h: set of local biases. Expects 1D np array.
        *** J: coupling strengths. Expects 2D np array of shape (len(s),len(s)).
        
        Optional Params:
        *** spin_val: (set) set of possible spin values
        *** create_spin_space (default = True): flag that decides whether or not to create the spin space
        *** name (default = ""): (str) name of the graph
        *** debug (default = False): (bool) instantiate graph in debug mode or not
        *** **kwargs: other parameters, used in multi-inheritance situations. Passed on to next class with super().

        MANDATORY METHODS
          - sign
          - strict_triu
        '''
        self._debug = debug
        self._h = h if type(h) == np.ndarray else np.array(h) # force h to be numpy array
        self._J = strict_triu(J) # throw away lower diagonal and diagonal
        if create_spin_space:
            self._spin_space = list(itertools.product(*[spin_val for i in range(0,len(h))]))
        self._in_notebook = is_notebook if is_notebook != -1 else IN_NOTEBOOK
        self._H = self.gen_tot_ham()
        self._ec = self.gen_energy_cosets()
        if self._debug:
            print("Running in notebook: {0}".format(self._in_notebook))
        if create_vis:
            self.create_network()
        super().__init__(**kwargs) # pass on unused keywords

### PROPERTIES ###
    @property
    def h(self):
        return self._h
    
    @property
    def J(self):
        return self._J
    
    @property
    def H(self):
        return self._H
    
    @property
    def size(self):
        return np.size(self.h)
    
    @property
    def ec(self):
        return self._ec

### Creation Methods ###
    def empty():
        ''' Creates empty IGraph instance

        Returns: G (IGraph)
        '''
        
        return IGraph(h=np.array([]),J=np.array([]))

### SEPARATE SETTER FUNCTIONS ###
    def set_h(self,i,hi):
        ''' Sets a specific index of h
        
        Returns: void

        Params:
        *** i: (int) the index of h to change
        *** hi: (float) the value h[i] will be changed to
        '''
        self._h[i] = hi
    
    def set_J(self,i,j,Jij):
        ''' Sets a specific index of h
        
        Returns: void

        Params:
        *** i: (int) the row entry of J to change
        *** j: (int) the column entry of J to change
        *** Jij: (float) the value J[i,j] will be changed to
        '''

        self._J[i,j] = Jij

### HAMILTONIAN METHODS ###  
    # ham calculation
    def calc_ham(s,h,J):
        ''' Calculates the Hamiltonian for the given spin (s) and parameters h and J. Assumes J is an upper triangular square matrix whose diagonal entries are all zero.
        
        Returns: (float) the Hamiltonian value

        Params:
        *** s: (numpy ndarray) the spin
        *** h: (numpy ndarray) the local biases
        *** J: (numpy ndarray) the coupling strengths
        '''

        if check_numpy(s) == False:
            s = np.array(s)
        if check_numpy(h) == False:
            h = np.array(h)
        if check_numpy(J) == False:
            J = np.array(J)
            
        # set Ham to zero and calculate it term by term
        Ham = 0
        for i in range(0, len(s)):
            # add all h contributions
            Ham += s[i]*h[i] 

            # add J contributions
            for j in range(i+1,len(s)):
                Ham += J[i,j]*s[i]*s[j]
        
        return Ham
    
    # get hamiltonian of this instance
    def get_ham(self,s):
        ''' Gets the Hamiltonian value for the spin s.

        Returns: the appropriate hamiltonian value for the given spin

        Params:
        *** s: (numpy ndarray) the spin whose Hamiltonian value is to be calculated.
        '''
        
        if self._debug:
            print("get_ham called")
            print("  - ham calculation:",IGraph.calc_ham(s,self.h,self.J))
            
        return IGraph.calc_ham(s,self.h,self.J)
   
    def gen_tot_ham(self,sort=True): 
        ''' Generates a dictionary whose keys are spins and whose values are the Hamiltonian values.
        
        Returns: H, a shape (n,2) array where n = number of vertices. 0th element of each row 
                     is spin input, 1st element is  corresponding Hamiltonian value.

        Optional Params:
        *** sort (default = True): (bool) flag which determines whether H is sorted after generation or not.
        '''
        
        H = []
        for spin in self._spin_space:
            if self._debug:
                print("Example row:",[spin,self.get_ham(np.array(spin))])
                
            H.append([spin,self.get_ham(np.array(spin))])
        
        H.sort(key=lambda row : row[len(row)-1])
        return H
     
    def display_ham(self,title=''):
        ''' Displays the complete Hamiltonian of this IGraph, as long as it has been generated.
        '''

        print(tabulate(self._H,headers=["Spin","Hamiltonian"]))
        
### ENERGY PARTITION/EQUIVALENCE CLASS METHODS ###
    def gen_energy_cosets(G):
        ''' Generates the energy cosets of G.

        Params:
        *** G: (IGraph) the Ising graph whose energy cosets are to be generated.
        '''

        ec = {}
        for row in G.H:
            if row[1] not in ec:
                ec[row[1]] = [row[0]]
            else:
                ec[row[1]].append(row[0])
        return ec
    
    def display_energy_cosets(G):
        ''' Displays a table of G's energy cosets.

        Params:
        *** G: (IGraph) the Ising graph whose energy cosets are to be displayed.
        '''

        table = [["Hamiltonian Value","Associated Spins"]] # the headers of the table

        # iterate through the energy cosets
        for key,value in G.ec.items():
            table.append([key,value])

        # display the table and put the total number of cosets at the bottom
        print(tabulate(table))
        print("Total # of Cosets: {0}".format(len(G.ec)))
        
    def display_param_ordering(G,hJ_format=False):
        ''' Generates a table illustrating the ordering of the parameters.

        Params:
        *** G: (IGraph) an instance of IGraph.

        Optional Params: 
        *** hJ_format (default = False): (bool) a flag which decides where to use A_ij format or h_i and J_ij format.
        '''


        params = G.get_sorted_labeled_abs_params()
        out = "0"
        oA = 0
        for k in range(len(params)):
            i,j,A = params[k]
            letter = "({0},{1})".format(i,j)
            if hJ_format:
                if i == j:
                    letter = "h_{0}".format(i)
                else:
                    letter = "J_{0},{1}".format(i,j)
            if A == oA: 
                out += " = {1}{0}".format(letter,sign(A,letter=True))
            else:
                out += " < {1}{0}".format(letter,sign(A,letter=True)) 
            oA = A
        print(out)
    
    def get_abs_order(G):
        ''' Creates an array containing all the h and J values, sorted. Does not remember which entries are h's and which entries are J's.

        Params:
        *** G: (IGraph) The Ising graph whose h and J parameters are to be sorted.
        '''

        A = []
        for i in range (G.size):
            A.append(abs(G.h[i])) # add the h[i] value
            for j in range(i+1,G.size):
                A.append(abs(G.J[i,j]))
                     
        A.sort()
        return A
    
    def get_params_mat(G):
        ''' Combines the h and J parameters of G into an upper triangular square matrix. The h values lie on the diagonal while the J values occupy the upper half diagonal.

        Params:
        *** G: (IGraph) The IGraph instance who's h and J parameters are to be combined.
        '''
        
        # create a numpy array of the appropriate size
        A = np.zeros((G.size,G.size))
        for i in range(G.size):
            
            # record the h_i value
            A[i,i] = G.h
            for j in range(i+1,G.size):
                
                # record the J_ij
                A[i,j] = G.J[i,j]

        return A
    
    def get_sorted_labeled_abs_params(G):
        ''' Gets a sorted list of the parameters h, J. Differentiates h and J entries by storing their indices.

        Returns: a list of triples (i,j,A) where i <= j. 
            - If i = j, then A is the h_i value of G.
            - If i < j, then A is the J_ij value of G.

        Params:
        *** G: (IGraph) an instance of IGraph.
        '''

        params = []
        for i in range(G.size):
            for j in range(i,G.size):
                A = G.h[i] if i == j else G.J[i,j]
                params.append((i,j,A)) # only include index if h and J nonzero
        params.sort(key=lambda row : abs(row[2]))
        
        return params
             
### GRAPH COMPARISON ###

    def compare_graphs(G1,G2,view=True):
        ''' A helpful method for comparing IGraphs.
            - Checks for equivalence between G1 and G2
            - Prints a table of two graphs and their Hamiltonian values across all spins in spin_space.
            - Requires that G1 and G2 have the same number of vertices.
        
        Returns: (bool) True if G1 and G2 are equivalent, False otherwise.

        Params:
        *** G1: (IGraph) First graph
        *** G2: (IGraph) Second graph

        Optional Params:
        *** view (default = True): (bool) view graphs as table
        '''

        value = True
        if G1.size != G2.size:
            print("Provided graphs have different number of vertices!")
            return False
        else:
            table = [["Spin #","Spin Graph 1","=?=","Spin Graph 2", "Hamiltonian 1","Hamiltonian 2"]]
            for i in range(len(G1.H)):
                a = G1.H[i][0]
                b = G2.H[i][0]
                if view:
                    table.append([i,G1.H[i][0],"=" if a == b else "=!=",G2.H[i][0],G1.H[i][1],G2.H[i][1]])
                    
                # set return value to false if graphs have unequal energy partitions
                if value and G1.H[i][0] != G2.H[i][0]:
                    value = False

            if view: print(tabulate(table))
            return value
        
    def check_equiv(G1,G2,view=False):
        value = True
        if G1.size != G2.size:
            if view:
                print("Provided graphs have different number of vertices!")
            return False
        else:
            if view:
                table = [["Spin #","Spin Graph 1","Spin Graph 2", "Hamiltonian 1","Hamiltonian 2"]]
            for i in range(len(G1.H)):
                a = G1.H[i][0]
                b = G2.H[i][0]
                if view:
                    print("{0} {2} {1}".format(a,b,"=" if a == b else "!="))
                    table.append([i,G1.H[i][0],G2.H[i][0],G1.H[i][1],G2.H[i][1]])
                    
                # set return value to false if graphs have unequal energy partitions
                if G1.H[i][0] != G2.H[i][0]:
                    value = False
                    # return early if not constructing viewable table
                    if view == False:
                        break
                 
            if view: print(tabulate(table))
            return value

### GRAPH GENERATION ###
    def gen_random_graph(size: int, low: int, high: int, all_int: bool = False):
        ''' Generates a random IGraph of the given size
        Returns: (IGraph) a random IGraph

        Params:
        *** size: (int) the size of the graph to generate
        *** low: smallest possible value for h and J
        *** high: highest possible value for h and J

        Optional Params:
        *** all_int (default = False): ensure all h and J values are integers
        '''

        if all_int:
            h = np.random.randint(low=low, high=high, size=size)
            J = np.random.randint(low=low, high=high, size=(size,size))
            
        else:
            h = np.random.rand(1,size)*(high - low) + low
            J = np.random.rand(size,size)*(high - low) + low 
            
        J = np.triu(J) - np.diag(np.diag(J)) # make J strictly upper triangular
        return IGraph(h = h, J = J)
#########################################

class PICircuit:
    ''' Preising Circuit Class
    '''
    
### static variables ###

    pre_defined_circ = ['AND', 'XOR','MUL'] # list of predefined circuit types
    
### Initializer ###

    def __init__(self,N: int, M: int, A: int = 0, *,name='',logic=None,logic_base=None,spin_val=[-1,1],in_spins=None,out_spins=None,aux_spins=None,make_logic_dict=False,**kwargs):
        ''' Initializer
        Params:
        *** N: (int); number of inputs vertices
        *** M: (int); number of outputs vertices

        Optional Params:
        *** A: (int) = 0; number of auxiliary vertices
        *** name (default = ''): (str) name of this PICircuit
        *** logic: (method) = None; method which returns spin of size M+A for every input of size N
        *** logic_base (default = None): (method) method which returns only output spin without auxiliary array.
        *** spin_val: (array) = [-1,1]; list of possible spin values
        *** in_spins (multidimensional array) space of input spins. Pass reference to conserve memory if possible (default None). 
        *** out_spins: (multidimensional array) space of output spins. Pass reference to conserve memory if possible (default None).
        *** aux_spins: (multidimensional array) space of auxiliary spins. Pass reference to conserve memory if possible (default None).
        *** make_logic_dict: (bool) Convert logic to dictionary on instantiation (default False).
        *** **kwargs: other parameters, used in multi-inheritance situations. Passed on to next class with super().
        '''
        
        self._N = N
        self._M = M
        self._A = A
        self._name = kwargs.get('name') if 'name' in kwargs else ''
        self._logic = logic
        self._logic_base = logic_base
        self._in_spins = get_spin_space(N,spin_val = spin_val) if in_spins == None else in_spins
        self._out_spins = get_spin_space(M,spin_val = spin_val) if out_spins == None else out_spins
        self._aux_spins = get_spin_space(A,spin_val = spin_val) if aux_spins == None else aux_spins
        self._logic_dict = self.logic_to_dict() if make_logic_dict else None


        #super().__init__(**kwargs)
    
### Properties, getters, setters ###

    def get_N(self): return self._N
    def set_N(self,N: np.ndarray): self._N = N
    N = property(get_N,set_N)
     
    def get_M(self): return self._M
    def set_M(self,M: np.ndarray): self._M = M
    M = property(get_M,set_M)

    def get_A(self): return self._A
    def set_A(self,A: np.ndarray): self._A = A
    A = property(get_A,set_A)

    @property
    def size(self):
        return self.N + self.M + self.A

    @property
    def name(self):
        return self._name

    def set_size(N = None, M = None, A = None):
        ''' Sets the size of the graph
        
        Optional Params:
        *** N (default = None): (int) the input size
        *** M (default = None): (int) the output size
        *** A (default = None): (int) the auxiliary size
        '''

        self.N = self.N if N == None else N
        self.M = self.M if M == None else M
        self.A = self.A if A == None else A

### Methods For Retrieving Circuit Values ###
    
    def get_value(self,ispin):
        ''' Returns the output of the circuit given an input. Uses logic dictionary if available, otherwise uses provided method.
        Returns: numpy array of length M + A, i.e. includes an auxiliary array on output

        Params:
        *** ispin: (numpy.ndarray) an input spin of size N
        '''

        out = self._logic_dict[ispin] if self._logic_dict != None else self._logic(ispin) # allows for logic to be a dictionary

        # ensure output is a numpy array
        if type(out) != np.ndarray:
            out = np.array(out)

        return out
    
    def get_value_base(self,ispin):
        ''' Returns only the output spin of the circuit with no auxiliary array. If not explicitly set, then will just truncate value from get_value().

        Returns: (numpy.ndarray) 1D length self.M

        Params:
        *** ispin: (numpy.ndarray) an input spin of size N
        '''

        if self._logic_base == None:
            return self.get_value(ispin)[:self.M]

        else:
            return self._logic_base(ispin)

####################################################################

    def get_all_values(self):
        ''' Produce a list of all correct outputs, i.e. get the image of the logic function in total output space

        Returns: (array) array consisting of all correct outputs to the problem
        '''

        outs = []
        for ispin in self._in_spins:
            outs.append(self.get_value(ispin))

        return outs

####################################################################

    def get_io_pairs(self,decimal=False):
        ''' Generates a list of all correct input/output pairs including auxiliary array in output.

        Returns: An array of length N containing spins of length N+M which correspond to 

        Optional Params:
        *** decimal (default=False): (bool) return spins in integer representation or not. If True, will convert in/out spin to a single integer.
        '''
        
        pairs = []
        for ispin in spinspace.get_spin_space(self.N):
            # get in/out spin
            spin = np.concatenate((ispin, self.get_value(ispin)))
            
            # add value to pairs array, either as a base10 number or a spin
            if decimal:
                pairs.append(spinspace.spin2dec(spin))
            else:
                pairs.append(spin)

        return pairs

####################################################################

    def get_io_pairs_base(self,decimal=False):
        ''' Produce list of all correct in/out pairs without auxiliary spins.
        
        Returns: (list) a list of spins of length N+M.

        Optional Params:
        *** decimal (default=False): (bool) return spins in integer representation or not. If True, will convert in/out spin to a single integer.
        '''

        pairs = []
        for ispin in spinspace.get_spin_space(self.N):
            # get in/out spin
            spin = np.concatenate((ispin, self.get_value_base(ispin)))
            
            # add value to pairs array, either as a base10 number or a spin
            if decimal:
                pairs.append(spinspace.spin2dec(spin))
            else:
                pairs.append(spin)

        return pairs

####################################################################

    def get_input_level(self,*ispins,display=False):
        '''Returns a dictionary keyed by the provided input spins and valued by a pair (out1,[outs]) where out1 is the correct output and [outs] is an array containing all incorrect outs for the level. All outputs have auxiliary array, if self.A > 0, attached to them.

        Params:
        *** *ispins: (numpy.ndarray) collection of input spins for which to fetch the levels
        *** display: (bool) display level as table (default False)
        '''

        levels = {}
        for ispin in ispins:
            ospin = self.get_value(ispin)
            levels[ispin] = [ospin,spinspace.get_spin_space(self.M+self.A)]
            levels[ispin][1].remove(ospin)

        return levels

####################################################################

    def get_input_level_base(self,*ispins,cat=False,display=False):
        '''Returns a dictionary keyed by the provided input spins and valued by a pair (out1,[outs]) where out1 is the correct output and [outs] is an array containing all incorrect outs for the level. No auxiliary arrays are attached to output.

        Params:
        *** *ispins: (numpy.ndarray) collection of input spins for which to fetch the levels
        *** cat (default=False): (bool) concatenate outputs with inputs
        *** display (default=False): (bool) display level as table
        '''

        if cat:
            levels = []
            for ispin in ispins:
                ospin = self.get_value_base(ispin)
                badouts = spinspace.get_spin_space(self.M)
                # remove the correct output from the set of wrong outputs
                # only works because the index of ospin in outs happens
                # to be the integer representation of ospin interpreted
                # as a binary number
                levels.append([np.concatenate((ispin,ospin)),
                               [np.concatenate((ispin,out)) for out in badouts]])
                # have to remove ospin here, as badouts is a reference to
                # a spinspace object we wish to persist.
                levels[len(levels)-1][1].pop(spinspace.spin2dec(ospin))

        else:
            levels = [0 for x in ispins]
            for ispin in ispins:
                ospin = self.get_value_base(ispin)
                levels[spinspace.spin2dec(ispin)] = [ospin,spinspace.get_spin_space(self.M)]
                levels[spinspace.spin2dec(ispin)][1].remove(ospin)

        return levels

####################################################################

    def min_norm(self,ispin):
        ''' Chooses the auxiliary spin that minimizes ising norm of the input/output/aux total spin
        '''

        import distance as dist
        
        # total number of parameters
        unit_param_num = 2**num_params(self.size)
        
        # throw away the auxiliary bit
        ospin = self.get_value_base(ispin)
        spin = np.concatenate((ispin,ospin))

        # initialize the min h, J, auxiliary spin and norm
        minh,minJ = get_unit_param(0,self.size)
        min_aspin = get_spin_space(self.A)[0]
        min_norm = dist.ising_norm(np.concatenate((spin,min_aspin)),minh,minJ)

        # the number here will correspond to 
        for num in range(unit_param_num):
            h,J = get_unit_param(num=num, G = self.size)
            for aspin in spinspace.get_spin_space(self.A):
                # attach the auxiliary to the spin
                tot_spin = np.concatenate((spin,aspin)) 
                norm = dist.ising_norm(tot_spin,h,J)

                # reassign everything if new minimum is found
                if norm < min_norm:
                    minh = h
                    minJ = J
                    min_aspin = aspin
                    min_norm = norm
                    print(f"h = {h}\nJ = {J}\nnorm = {norm}")
        
        return (np.concatenate((spin,min_aspin)),minh,minJ,min_norm)

####################################################################

    def min_separation_base(self):
        ''' Finds the unit h and J which maximize the separation of valid in/out pairs from wrong in/out pairs across all levels
        '''

        import distance as dist
        
        # total number of parameters
        unit_param_num = 2**num_params(self.N + self.M)
        
        # initialize the min h, J, auxiliary spin and norm
        maxh,maxJ = get_unit_param(0,self.size)

        maxD = 0
        D = maxD
        for num in range(unit_param_num):
            # get the h and J for this round
            h,J = get_param_mod3(num=num, G = self.N+self.M)
            for ispin in spinspace.get_spin_space(self.N):
                # get valid output and bad outputs for current input
                out,bad_outs = self.get_input_level_base(ispin,cat=True)[0]
                D += dist.avg_dist_from_center(out,
                                               bad_outs,
                                               h,
                                               J,
                                               distkey='signed-ising')
                if D > maxD:
                    maxD = D
                    maxh = h
                    maxJ = J

        return (maxh,maxJ,maxD)
            
    
####################################################################

### Methods for Comparing Spin Values ###
    def isCorrect(self,ispin,ospin):
        ''' Checks whether ospin mathes the output of ispin. Throws away auxiliary values.

        Returns: True if ispin --> ospin, False otherwise

        Params:
        *** ispin: (numpy.ndarray) input spin, length N
        *** ospin: (numpy.ndarray) output spin, length >= M. If auxiliary array is here then it is thrown away for the check
        '''

        return np.array_equal(self.get_value(ispin)[:self.M],ospin[:self.M])

### Display Methods ###

    def display_circuit(self,*other_methods,headers=[]):
        ''' Displays a table of this circuit. Displays valid input/output pairs by default, can display additional columns with entries generated by *other_methods.

        Returns: nothing

        Optional Params:
        *** *other_methods: other methods used to populate additional columns in the table. Methods must take only two positional arguments, ispin and ospin, corresponding to input and output spins.
        *** headers (deafult = []): (array(str)) used to input additional headers labeling the columns populated by user-defined *other_methods

        '''
        
        from tabulate import tabulate

        a = [] # the table of things to display
        main_headers = ["Input","Output"] + (["Auxiliary"] if self.A > 0 else []) # headers of the table
        for ispin in self._in_spins:  # append all circuit values to the table
            ospin = self.get_value(ispin)
            temp = [ispin, ospin[:self.M]]
            if self.A > 0: 
                temp = temp + [ospin[self.M:self.M+self.A]] # append only the auxiliary
            a.append( temp + [method(ispin,ospin) for method in other_methods] )
                     
        print(tabulate(a,main_headers + headers))

### Spin Action Methods ###
    
### Creation Methods ###

    def empty():
        ''' Returns an empty instance of PICircuit

        Returns: G (PICircuit) instance
        '''

        return PICircuit(N = 0, M = 0)

### PICircuit Meta Methods ###

    def logic_to_dict(self):
        ''' Creates self._logic_dict, a dictionary containing the logic of the circuit. 
        '''

        # the logic dictionary
        self._logic_dict = {}
        
        # make the dictionary
        for ispin in self._in_spins:
            self._logic_dict[ispin] = self._logic(ispin)

#########################################

class ICircuit(PICircuit,IGraph):
    ''' Class for an Ising circuit.
    '''

    def __init__(self,h,J,N,M,A=0,spin_val=[-1,1],in_spins=None,out_spins=None,aux_spins=None,make_logic_dict=False):
        ''' Initializer
        Params:
        *** h: (numpy.ndarray) the local biases
        *** J: (numpy.ndarray) the coupling strengths
        *** N: (int); number of inputs vertices
        *** M: (int); number of outputs vertices

        Optional Params:
        *** A: (int) = 0; number of auxiliary vertices
        *** spin_val: (array) = [-1,1]; list of possible spin values
        *** in_spins (multidimensional array) space of input spins. Pass reference to conserve memory if possible (default None). 
        *** out_spins: (multidimensional array) space of output spins. Pass reference to conserve memory if possible (default None).
        *** aux_spins: (multidimensional array) space of auxiliary spins. Pass reference to conserve memory if possible (default None).
        *** make_logic_dict: (bool) Convert logic to dictionary on instantiation (default False).
        '''

        if N + M + A != len(h):
            print("THROW ERROR")

        # call initializers of PICircuit and IGraph
        super().__init__(N=N,M=M,A=A,h=h,J=J,logic= self._gamma,spin_val=spin_val,in_spins=in_spins,out_spins=out_spins,aux_spins=aux_spins,make_logic_dict=make_logic_dict,create_spin_space=False)

### Creation Methods ###

    def empty():
        ''' Creates empty IGraph instance

        Returns: G (IGraph)
        '''
        
        return IGraph(N=0,M=0,h=np.array([]),J=np.array([]))

### Hamiltonian and Circuit Methods ###

    def _gamma(self,ispin):
        ''' Logic of an Ising Circuit. Gets the output which minimizes the Hamiltonian.
        
        Returns: out_spin (numpy.ndarray) length M+A numpy array which minimizes the Hamiltonian for the given input.

        Params:
        *** ispin: (numpy.ndarray) input spin of length N
        '''

        # create dict of output/hamiltonian values
        d = {ospin : self.calc_ham(ispin,ospin) for ospin in self._out_space} 

        # return output spin which minimizes hamiltonian
        return min(d, key=d.get)
    
    def calc_ham(self,ispin,ospin):
        ''' Overloads the calc_ham function provided by IGraph. Changes arguments to accept two positional arguments instead of three, an input and output spin. Used in ICircuit.display_circuit to populate third column of cicuit table with Hamiltonian values.

        Returns: (float) Hamiltonian value for given ispin and ospin pair

        Params:
        *** ispin: (numpy.ndarray) the input spin, length N
        *** ospin: (numpy.ndarray) the output spin, length M + A
        '''
        
        return IGraph.calc_ham(np.append(ispin,ospin), self.h, self.J)

### Visualization Methods ###
    def display_circuit(self):
        ''' Overloads the PICircuit.display_circuit function. Adds a third column to the table displaying the Hamiltonian value of the input/output pair.
        '''

        super().display_circuit(self.calc_ham,headers=["Hamiltonian"])

#########################################

class IMul(PICircuit):
    ''' Multiplication preising circuit
    '''

    def __init__(self,N1,N2,A,**kwargs):
        '''
        Description: initialize a multiplication circuit.
          N1: number of inputs for first number
          N2: number of inputs for second number
          A: number of auxiliary spins
        '''
        self._N1 = N1
        self._N2 = N2
        self._A = A
        super().__init__(N=N1+N2, M=N1+N2, logic=self.mult_logic_w_aux, A=A,store_logic=True,name=f"MUL_{N1}x{N2}x{A}",**kwargs)
        
### GETTERS AND SETTERS ###

    def get_N1(self): return self._N1
    def set_N1(self,N1: int): 
        self._N1 = N1
        self.N = N1 + N2
    N1 = property(get_N1,set_N1)
    
    def get_N2(self): return self._N2
    def set_N2(self,N2: int): 
        self._N2 = N2
        self.N = N1+N2
    N2 = property(get_N2,set_N2) 
    
    def get_A(self): return self._A
    def set_A(self,A: int): self._A = A
    A = property(get_A,set_A) 

### INSTANCE METHODS ###

    # multiplication logic
    def mult_logic(self,ispin):
        out_len = self.N1 + self.N2
        num1 = int(''.join(['0' if x == -1 else str(x) for x in ispin[:self.N1]]),2)
        num2 = int(''.join(['0' if x == -1 else str(x) for x in ispin[self.N1:out_len]]),2)
        return np.array([-1 if x == 0 else x for x in mult_to_binary(num1,num2,length=out_len)])
    
    def mult_logic_w_aux(self,ispin):
        '''
        This will need to be changed once new results are added, simply sets all aux spins to one
        '''
        return np.append(self.mult_logic(ispin), np.ones(self.A)) 
        
    # add functionality to display_circuit
    def display_circuit(self):
        super().display_circuit(self.spins_to_decimal_prod, headers=['Decimal Rep'])
        
    def spins_to_decimal_prod(self,ispin,ospin):
        num1 = int(''.join(['0' if x == -1 else str(x) for x in ispin[:self.N1]]),2)
        num2 = int(''.join(['0' if x == -1 else str(x) for x in ispin[self.N1:]]),2)
        out = int(''.join(['0' if x == -1 else str(int(x)) for x in ospin[:self.M - self.A]]),2)
    
    def min_aux_choice(self,ispin):
        ''' Chooses the auxiliary spin that minimizes ising norm of the input/output/aux total spin
        '''

        import distance as dist

        return np.append(self.mult_logic(ispin), np.ones(self.A)) 

        
####################################################################

class IXOR(PICircuit):
    ''' XOR preising circuit
    '''

    def __init__(self,N1,A=0,**kwargs):
        '''
        Description: initialize a multiplication circuit.
          N: size of one input
          A: number of auxiliary spins
        '''
        self._N1 = N1
        self._A = A
        super().__init__(N=2*N1, M=N1, logic=self.xor_logic_w_aux, A=A,store_logic=True,name=f"XOR_{N1}x{N1}x{A}",**kwargs)
        
### GETTERS AND SETTERS ###

    def get_N1(self): return self._N1
    def set_N1(self,N1: int): 
        self._N1 = N1
        self.N = N1*2
    N1 = property(get_N1,set_N1) 
 
    def get_A(self): return self._A
    def set_A(self,A: int): self._A = A
    A = property(get_A,set_A) 

### INSTANCE METHODS ###

    # multiplication logic
    def xor_logic(self,ispin):
        ''' Logic of the xor circuit
        '''

        # pointwise multiply and then multiply by -1
        return np.multiply(ispin[:self.N1],ispin[self.N1:])*(-1)
    
    def xor_logic_w_aux(self,ispin):
        '''
        This will need to be changed once new results are added, simply sets all aux spins to one
        '''
        return np.append(self.xor_logic(ispin), np.ones(self.A)) 
        
### DISPLAY METHODS ### 
    # add functionality to display_circuit
    def display_circuit(self):
        super().display_circuit()
