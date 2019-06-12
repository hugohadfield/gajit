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
    def test_something(self):

        def example_one(X, R):
            Y = 1 + (R * X * (~R))
            D = Y * (e1 ^ e2 ^ e3 ^ einf ^ e0)
            F = D | (e1 ^ e2)
            return F

        @gajit([layout.grade_mask(1), layout.rotor_mask], galgebra=True)
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



if __name__ == '__main__':
    unittest.main()
