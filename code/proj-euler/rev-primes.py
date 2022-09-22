import math
from tabulate import tabulate

def rev(num):
    rev = 0
    while(num > 0):
        rev = rev * 10 + num % 10
        num = num // 10

    return rev

def is_prime(p):
    """ Checks whether input is prime or not.
    Returns: bool

    Params:
    *** p: (int) the number whose primality is in question
    """

    N = math.ceil(math.sqrt(p))
    prime = True

    for d in range(2,N):
        if p % d == 0:
            prime = False
            break

    return prime


def primes_in_range(min,max):
    """ Returns the primes p such that min <= p <= max
    """

    primes = []
    for p in range(min,max+1):
        if is_prime(p):
            primes.append(p)

    return primes

def get_rps(min,max):
    """ Get the reversible prime squares corresponding to primes between min and max
    Returns: list (int)
    """
    primes = primes_in_range(min,max)
    rps = []
    table = []

    for p in primes:
        n1 = p**2
        n2 = rev(n1)
        
        # ignore palindromes
        if n1 == n2:
            continue

        # ignore if q if not a perfect square
        q = int(math.sqrt(n2))
        if q**2 != n2:
            continue
        
        if is_prime(q) == False:
            continue

        else:
            table.append([p,n1,n2,q])
            rps.append(n1)

    return rps,table

if __name__ == '__main__':
    
    ans = 0
    num = 0
    step = 10**5
    table = []
    full_rps = []
    for min in range(0,10**8,step):
        max = min + step
        rps,tab = get_rps(min,max)
        full_rps = full_rps + rps
        table = table + tab
        num += len(rps)
        ans += sum(rps)
        print(f"Found {num} reversible prime squares, sum is {ans}.")
        print(f"   ...now searching for primes between {min} and {max}...")
        
        if num == 50:
            break
    print(tabulate(table,headers=["p","n1","n2","q"])) 
    print(f"Found {num} reversible prime squares, sum: {ans}")
