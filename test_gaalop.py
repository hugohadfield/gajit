
import os
os.environ['GAALOP_CLI_HOME'] = '/work/Gaalop/target/gaalop-1.0.0-bin'

from gaJIT import *

import unittest
from clifford.tools.g3c import *
from clifford.g3c import *
import time




@gaJIT(mask_list=[GradeMasks.onevectormask.value, GradeMasks.threevectormask.value], verbose=1)
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

class TestJit(unittest.TestCase):
    def test_jit_example(self):
        for i in range(1000):
            P = random_conformal_point()
            C = random_circle()

            np.testing.assert_almost_equal(pp_algo(P, C).value, pp_algo_wrapped(P, C).value)

        start_time = time.time()
        for i in range(10000):
            pp_algo_wrapped(P, C)
        t_gaalop = time.time() - start_time
        print('GAALOP ALGO 2: ', t_gaalop)

        start_time = time.time()
        for i in range(10000):
            pp_algo(P, C)
        t_trad = time.time() - start_time
        print('TRADITIONAL 2: ', t_trad)

        print('T/G: ', t_trad / t_gaalop)
