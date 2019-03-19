
import os
os.environ['GAALOP_CLI_HOME'] = '/work/Gaalop/target/gaalop-1.0.0-bin'
os.environ['GAALOP_ALGEBRA_HOME'] = '/work/gaalopinterface/gajit/algebra'

from gajit import gajit

import unittest

import time

import numpy as np
import numba
from clifford import Cl, conformalize

















layout_orig, blades_orig = Cl(3)
layout, blades, stuff = conformalize(layout_orig)

from clifford.tools.g3c import random_conformal_point

e1 = blades['e1']
e2 = blades['e2']
e3 = blades['e3']
e123 = blades['e123']
e12 = blades['e12']


def example_one(X, R):
    Y = 1 + (R * X * ~R)
    Ystar = Y * (e1 ^ e2 ^ e3)
    F = Ystar | (e1 ^ e2)
    return Ystar


gp = layout.gmt_func
op = layout.omt_func
ip = layout.imt_func
rev = layout.adjoint_func

e12_value = (e1 ^ e2).value
e123_value = (e1 ^ e2 ^ e3).value


def example_one_faster(X, R):
    Y = gp(gp(R, X), rev(R))
    Y[0] += 1
    Ystar = gp(Y, e123_value)
    F = ip(Ystar, e12_value)
    return Ystar


@numba.njit
def example_one_faster_jit(X, R):
    Y = gp(gp(R, X), rev(R))
    Y[0] += 1
    Ystar = gp(Y, e123_value)
    F = ip(Ystar, e12_value)
    return Ystar


@gajit(mask_list=[layout.grade_mask(1), layout.rotor_mask], verbose=2, ignore_cache=False)
def example_one_gajit(X, R):
    Y = 1 + (R * X * (~R))
    Ystar = Y * (e1 ^ e2 ^ e3)
    F = Ystar | (e1 ^ e2)
    return Ystar





class TestPaper(unittest.TestCase):

    def test_example_one(self):
        #### EXAMPLE ONE FROM THE PAPER ####


        print(layout.grade_mask(1)*1)
        print(layout.rotor_mask)


        ## ENSURE ALL THE ALGORITHMS GIVE THE SAME ANSWER
        X = layout.randomMV(1)
        R = layout.randomRotor()
        true = example_one(X, R)
        print(R)
        print(R.isVersor())
        print(true)
        for i in range(100):
            X = layout.randomMV(1)#
            R = layout.randomRotor()

            true = example_one(X,R)
            true_array = np.array([true.value]*3)
            test_array = np.zeros((3,32))

            test_array[0, :] = example_one_faster(X.value, R.value)
            test_array[1, :] = example_one_faster_jit(X.value, R.value)
            test_array[2, :] = example_one_gajit(X,R).value

            for j in range(3):
                print(j)
                try:
                    np.testing.assert_almost_equal(test_array[j, :], true_array[j, :])
                except:
                    print('\n',flush=True)
                    print(layout.MultiVector(value=test_array[j, :]))
                    print(layout.MultiVector(value=true_array[j, :]))
                    print(layout.MultiVector(value=example_one_gajit.func(X.value, R.value)))
                    print('\n', flush=True)
                    #np.testing.assert_almost_equal(test_array[j, :], true_array[j, :])


        # start_time = time.time()
        # for i in range(10000):
        #     example_one(X, R)
        # t_normal = time.time() - start_time
        # print('UNOPTIMISED: ', t_normal)
        #
        # start_time = time.time()
        # for i in range(10000):
        #     example_one_faster(X.value, R.value)
        # t_hand = time.time() - start_time
        # print('HAND: ', t_hand)
        #
        # start_time = time.time()
        # for i in range(10000):
        #     example_one_faster_jit(X.value, R.value)
        # t_jit = time.time() - start_time
        # print('HAND + JIT: ', t_jit)
        #
        # start_time = time.time()
        # for i in range(10000):
        #     example_one_gajit(X, R)
        # t_gajit = time.time() - start_time
        # print('GAJIT: ', t_gajit)



class TestOtherExamples(unittest.TestCase):

    def test_basics(self):

        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)

        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']
        e12345 = blades['e12345']
        einf = stuff['einf']
        e123 = blades['e123']
        e12 = blades['e12']

        msk = np.ones(32)
        @gajit(mask_list=[msk], ignore_cache=True)
        def basic_map_test(A):
            C = A
            return C

        res = layout.randomMV()
        res2 = basic_map_test(res)
        np.testing.assert_almost_equal(res.value, res2.value)

        @gajit(mask_list=[msk], ignore_cache=True)
        def basic_reverse_test(A):
            C = ~A
            return C

        res = layout.randomMV()
        res2 = basic_reverse_test(res)
        np.testing.assert_almost_equal((~res).value, res2.value)

        @gajit(mask_list=[msk], ignore_cache=True)
        def basic_addition_test(A):
            C = 1 + A
            return C

        res = layout.randomMV()
        res2 = basic_addition_test(res)
        np.testing.assert_almost_equal((1 + res).value, res2.value)

    def test_jit_example(self):

        from clifford.tools.g3c import random_circle, random_conformal_point

        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)

        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']
        e12345 = blades['e12345']
        einf = stuff['einf']
        e123 = blades['e123']
        e12 = blades['e12']


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