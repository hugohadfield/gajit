# gajit
A JIT compiler for geometric algebra in python

Using [Gaalop](http://www.gaalop.de/) and [numba](http://numba.pydata.org/) this package symbolically optimises geomteric algebra algorithms, generates numpy code and JITs the resultant code. **Currently only supports Cl(4,1)**, still very experimental and hacky. Currently produces around an order of magnitude speed up for normally written algorithms and around **2-4x speed up over hand optimised code**. Many further optimisations are possible that so far have not been implemented, Pull Requests welcomed.

# Example usage

The package comes in the form of a decorator that can be applied to normal functions written with the [clifford](https://www.github.com/pygae/clifford) GA syntax. Grade masks can be specified as arguements **this is highly recommended for both run and compile time speedups** verbosity of the JIT process can be adjusted for debugging:
```
from gajit import *
from clifford.g3c import *

@gajit(mask_list=[layout.grade_mask(1), layout.grade_mask(3)], verbose=1)
def pp_algo_wrapped(P, C):
    Cplane = C ^ einf
    Cd = C * (Cplane)
    L = (Cplane * P * Cplane) ^ Cd ^ einf
    pp = Cd ^ (L * e12345)
    return pp

def pp_algo(P, C):
    Cplane = C ^ einf
    Cd = C * (Cplane)
    L = (Cplane * P * Cplane) ^ Cd ^ einf
    pp = Cd ^ (L * e12345)
    return pp
```

The jitted function can be called as normal and **much faster** than the original
```
from clifford.tools.g3c import *
import time

P = random_conformal_point()
C = random_circle()

print( pp_algo_wrapped(P, C) )
print( pp_algo(P, C) )
   
start_time = time.time()
for i in range(10000):
    pp_algo_wrapped(P, C)
t_gaalop = time.time() - start_time
print('gaJIT ALGO: ', t_gaalop)

start_time = time.time()
for i in range(10000):
    pp_algo(P, C)
t_trad = time.time() - start_time
print('TRADITIONAL: ', t_trad)

print('T/G: ', t_trad / t_gaalop)
```

# Getting Gaalop
This requires Gaalop to be compiled in command line interface (cli) mode, instructions on setting this up are on the TODO list

# TODO
- [ ] Describe the Gaalop CLI setup and environment variables
- [ ] Provide a pip installable setup.py 
- [ ] Host on pypi
- [ ] Implement more transpilation of CLUSCRIPT language features
- [ ] Implement a non-mapped metric for speed + simplicity
- [ ] Implement a proper codegen cache system
- [ ] Add a blade mask inference option

