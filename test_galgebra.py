import os
os.environ['GAALOP_CLI_HOME'] = '/work/Gaalop/target/gaalop-1.0.0-bin'
os.environ['GAALOP_ALGEBRA_HOME'] = '/work/gaalopinterface/gajit/algebra'


import unittest
from gajit import *
import numba
from clifford.g3c import *
from clifford.tools.g3c import *
import time

e0 = eo


class MyTestCase(unittest.TestCase):
    def test_galgebra_example(self):

        def example_one(X, R):
            Y = 1 + (R * X * (~R))
            D = Y * (e1 ^ e2 ^ e3 ^ einf ^ e0)
            F = D | (e1 ^ e2)
            return F

        @gajit([layout.grade_mask(1), layout.rotor_mask], galgebra=True, ignore_cache=True)
        def example_one_galgebra(X, R):
            Y = 1 + (R * X * (~R))
            D = Y * (e1 ^ e2 ^ e3 ^ einf ^ e0)
            F = D | (e1 ^ e2)
            return F

        @gajit(mask_list=[layout.grade_mask(1), layout.rotor_mask], verbose=1, ignore_cache=True)
        def example_one_gajit(X, R):
            Y = 1 + (R * X * (~R))
            D = Y * (e1 ^ e2 ^ e3 ^ einf ^ e0)
            F = D | (e1 ^ e2)
            return F

        nruns = 10000

        for i in range(1000):

            A = random_conformal_point()
            B = random_rotation_translation_rotor()
            res = example_one(A, B)
            res2 = example_one_galgebra(A, B)
            res3 = example_one_gajit(A, B)
            np.testing.assert_almost_equal(res.value, res2.value)
            np.testing.assert_almost_equal(res.value, res3.value)

        start_time = time.time()
        for i in range(nruns):
            example_one(A, B)
        t_normal = time.time() - start_time
        print(1000*1000*t_normal / nruns)

        start_time = time.time()
        for i in range(nruns):
            example_one_galgebra(A, B)
        t_galgebra = time.time() - start_time
        print(1000*1000*t_galgebra / nruns)

        start_time = time.time()
        for i in range(nruns):
            example_one_gajit(A, B)
        t_galop= time.time() - start_time
        print(1000 * 1000 * t_galop / nruns)

    def test_galgebra_rot(self):
        def example_rot(L, M):
            K = 2 + M * L + L * M
            T = -(K | e0) * einf
            P = (2 * (K - T))[0]
            Y = 1 - (T / P)
            R = Y * (1 + M * L)
            return R

        @gajit([layout.grade_mask(3), layout.grade_mask(3)], galgebra=True, ignore_cache=True)
        def example_rot_galgebra(L, M):
            K = 2 + M * L + L * M
            T = -(K | e0) * einf
            P = (2 * (K - T))[0]
            Y = 1 - (T / P)
            R = Y * (1 + M * L)
            return R

        @gajit(mask_list=[layout.grade_mask(3), layout.grade_mask(3)], verbose=1, ignore_cache=True)
        def example_rot_gajit(L, M):
            K = 2 + M * L + L * M
            T = -(K | e0) * einf
            P = (2 * (K - T))
            Y = 1 - (T / P)
            R = Y * (1 + M * L)
            return R

        nruns = 10000

        for i in range(1000):
            A = random_line()
            B = random_line()
            res = example_rot(A, B)
            res2 = example_rot_galgebra(A, B)
            res3 = example_rot_gajit(A, B)
            try:
                np.testing.assert_almost_equal(res.value, res2.value)
                np.testing.assert_almost_equal(res.value, res3.value)
            except:
                print()
                print()
                print(res)
                print()
                print(res2)
                print()
                print(res3)
                print(flush=True)

                K = 2 + M * L + L * M
                T = -(K | e0) * einf
                P = (2 * (K - T))[0]
                Y = 1 - (T / P)
                R = Y * (1 + M * L)

                np.testing.assert_almost_equal(res.value, res2.value)
                np.testing.assert_almost_equal(res.value, res3.value)

        start_time = time.time()
        for i in range(nruns):
            example_rot(A, B)
        t_normal = time.time() - start_time
        print(1000 * 1000 * t_normal / nruns)

        start_time = time.time()
        for i in range(nruns):
            example_rot_galgebra(A, B)
        t_galgebra = time.time() - start_time
        print(1000 * 1000 * t_galgebra / nruns)

        start_time = time.time()
        for i in range(nruns):
            example_rot_gajit(A, B)
        t_galop = time.time() - start_time
        print(1000 * 1000 * t_galop / nruns)




if __name__ == '__main__':
    unittest.main()
