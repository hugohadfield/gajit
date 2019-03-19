
import os
os.environ['GAALOP_CLI_HOME'] = '/work/Gaalop/target/gaalop-1.0.0-bin'
os.environ['GAALOP_ALGEBRA_HOME'] = '/work/gaalopinterface/gajit/algebra'


from gajit import *

import unittest
from clifford.tools.g3c import *
from clifford.g3c import *
import time



#### A GENERAL EXAMPLE ####

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



#### EXAMPLE ONE FROM THE PAPER ####


def example_one(X, R):
    Y = 1 + R*X*~R
    Ystar =  Y*(e1^e2^e3)
    F = Ystar|(e1^e2)
    return F


gp = layout.gmt_func
op = layout.omt_func
ip = layout.imt_func
rev = layout.adjoint_func

e12_value = e12.value
e123_value = e123.value

def example_one_faster(X, R):
    Y = gp(gp(R,X),rev(R))
    Y[0] += 1
    Ystar = gp(Y,e123_value)
    F = ip(Ystar,e12_value)
    return F


@numba.njit
def example_one_faster_jit(X, R):
    Y = gp(gp(R,X),rev(R))
    Y[0] += 1
    Ystar = gp(Y,e123_value)
    F = ip(Ystar,e12_value)
    return F


@gajit(mask_list=[layout.grade_mask(1), layout.rotor_mask])
def example_one_gajit(X, R):
    Y = 1 + R*X*~R
    Ystar =  Y*(e1^e2^e3)
    F = Ystar|(e1^e2)
    return F


class TestJit(unittest.TestCase):

    def test_example_one(self):
        ## ENSURE ALL THE ALGORITHMS GIVE THE SAME ANSWER
        X = random_conformal_point()
        R = generate_dilation_rotor(np.random.rand() + 0.5) * random_rotation_translation_rotor()
        true = example_one(X, R)
        print(true)
        for i in range(1000):
            X = random_conformal_point()
            R = generate_dilation_rotor(np.random.rand()+0.5)*random_rotation_translation_rotor()

            true = example_one(X,R)
            true_array = np.array([true.value]*3)
            test_array = np.zeros((3,32))

            test_array[0, :] = example_one_faster(X.value, R.value)
            test_array[1, :] = example_one_faster_jit(X.value, R.value)
            test_array[2, :] = example_one_gajit(X,R).value

            np.testing.assert_almost_equal(test_array, true_array)


        start_time = time.time()
        for i in range(10000):
            example_one(X, R)
        t_normal = time.time() - start_time
        print('UNOPTIMISED: ', t_normal)

        start_time = time.time()
        for i in range(10000):
            example_one_faster(X.value, R.value)
        t_hand = time.time() - start_time
        print('HAND: ', t_hand)

        start_time = time.time()
        for i in range(10000):
            example_one_faster_jit(X.value, R.value)
        t_jit = time.time() - start_time
        print('HAND + JIT: ', t_jit)

        start_time = time.time()
        for i in range(10000):
            example_one_gajit(X, R)
        t_gajit = time.time() - start_time
        print('GAJIT: ', t_gajit)



    def test_jit_example(self):
        for i in range(1000):
            P = random_conformal_point()
            C = random_circle()

            np.testing.assert_almost_equal(pp_algo(P, C).value, pp_algo_wrapped(P, C).value)

        start_time = time.time()
        for i in range(10000):
            pp_algo_wrapped(P, C)
        t_gaalop = time.time() - start_time
        print('GAJIT ALGO: ', t_gaalop)

        start_time = time.time()
        for i in range(10000):
            pp_algo(P, C)
        t_trad = time.time() - start_time
        print('TRADITIONAL: ', t_trad)

        print('T/G: ', t_trad / t_gaalop)

if __name__ == '__main__':
    unittest.main()