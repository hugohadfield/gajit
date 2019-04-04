
import os
os.environ['GAALOP_CLI_HOME'] = '/work/Gaalop/target/gaalop-1.0.0-bin'
os.environ['GAALOP_ALGEBRA_HOME'] = '/work/gaalopinterface/gajit/algebra'

from gajit import gajit, Algorithm

import unittest

import time

import numpy as np
import numba
from clifford import Cl, conformalize


class TestPaper(unittest.TestCase):

    def test_ray_tracer(self):

        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)
        e12345 = blades['e12345']
        einf=stuff['einf']

        from clifford.tools.g3c import random_conformal_point, random_sphere

        # Write the body of a function as a string here
        body = """
        R = *(A ^ C ^ einf);
        PP = R ^ (*S);
        hasIntersection = PP.PP; 
        """

        # Create a python function from that GAALOPScript function
        raytrace = Algorithm(
            inputs=['S', 'A', 'C'],
            outputs=['hasIntersection'],
            intermediates=['R', 'PP'],
            body=body,
            function_name="raytracer",
            blade_mask_list=[layout.grade_mask(4),
                             layout.grade_mask(1),
                             layout.grade_mask(1)]
        )

        def clifford_raytrace(S,A,C):
            PP = ((e12345 * (A ^ C ^ einf)) ^ (e12345 * S))
            hasIntersection = (PP | PP)
            return hasIntersection

        @gajit([layout.grade_mask(4),layout.grade_mask(1),layout.grade_mask(1)])
        def clifford_raytrace_gajit(S,A,C):
            PP = ((e12345 * (A ^ C ^ einf)) ^ (e12345 * S))
            hasIntersection = (PP | PP)
            return hasIntersection

        for i in range(1000):
            A = random_conformal_point()
            C = random_conformal_point()
            S = random_sphere()

            res = raytrace(S,A,C)
            res2 = clifford_raytrace(S,A,C)
            res3 = clifford_raytrace_gajit(S,A,C)
            np.testing.assert_almost_equal(res2[0], res[0])
            np.testing.assert_almost_equal(res3[0], res[0])

        nruns = 10000

        start_time = time.time()
        for i in range(nruns):
            clifford_raytrace(S,A,C)
        t_normal = time.time() - start_time
        print('CLIFFORD: ', (10**6)*t_normal/nruns)

        start_time = time.time()
        for i in range(10000):
            raytrace(S,A,C)
        t_hand = time.time() - start_time
        print('GAALOPScript: ', (10**6)*t_hand/nruns)

        start_time = time.time()
        for i in range(10000):
            clifford_raytrace_gajit(S, A, C)
        t_hand = time.time() - start_time
        print('GAJIT: ', (10 ** 6) * t_hand / nruns)

    def test_ray_hit_circle(self):
        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)
        e12345 = blades['e12345']
        einf = stuff['einf']

        from clifford.tools.g3c import random_line, random_circle

        omt_func = layout.omt_func
        imt_func = layout.imt_func

        @gajit([layout.grade_mask(2), layout.grade_mask(2)])
        def dual_line_dual_circle_hit_gajit(dl, dc):
            t = (dc^dl)
            out = t|t
            return out

        def dual_line_dual_circle_hit(dl, dc):
            t = (dc^dl)
            out = t|t
            return out[0]

        def dual_line_dual_circle_hit_hand(dl, dc):
            t = omt_func(dc, dl)
            out = imt_func(t, t)
            return out[0]

        for i in range(1000):
            L = random_line()
            C = random_circle()

            dl = e12345 * L
            dc = e12345 * C

            res = dual_line_dual_circle_hit_gajit.func(dl.value, dc.value)[0]
            res2 = dual_line_dual_circle_hit(dl, dc)
            res3= dual_line_dual_circle_hit_hand(dl.value, dc.value)

            np.testing.assert_almost_equal(res, res2)
            np.testing.assert_almost_equal(res, res3)

        nruns = 10000

        L = random_line()
        C = random_circle()

        dl = e12345 * L
        dc = e12345 * C

        start_time = time.time()
        for i in range(nruns):
            dual_line_dual_circle_hit(dl, dc)
        t_normal = time.time() - start_time
        print('UNOPTIMISED: ', (10 ** 6) * t_normal / nruns)

        start_time = time.time()
        for i in range(nruns):
            dual_line_dual_circle_hit_gajit.func(dl.value, dc.value)[0]
        t_gajit = time.time() - start_time
        print('GAJIT: ', (10 ** 6) * t_gajit / nruns)

        start_time = time.time()
        for i in range(nruns):
            dual_line_dual_circle_hit_hand(dl.value, dc.value)
        t_hand = time.time() - start_time
        print('HAND: ', (10 ** 6) * t_gajit / nruns)


    def test_example_one(self):

        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)

        from clifford.tools.g3c import random_conformal_point

        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']
        e4 = blades['e3']
        e5 = blades['e3']
        e12345 = blades['e12345']
        e12 = blades['e12']

        einf = stuff['einf']
        e0 = stuff['eo']

        def example_one(X, R):
            Y = 1 + (R * X * ~R)
            Ystar = Y * (e1 ^ e2 ^ e3 ^ einf ^ e0)
            F = Ystar | (e1 ^ e2)
            return F

        gp = layout.gmt_func
        op = layout.omt_func
        ip = layout.imt_func
        rev = layout.adjoint_func

        e12_value = (e1 ^ e2).value
        e12345_value = (e1 ^ e2 ^ e3 ^ einf ^ e0).value

        def example_one_faster(X, R):
            Y = gp(gp(R, X), rev(R))
            Y[0] += 1
            Ystar = gp(Y, e12345_value)
            F = ip(Ystar, e12_value)
            return F

        @numba.njit
        def example_one_faster_jit(X, R):
            Y = gp(gp(R, X), rev(R))
            Y[0] += 1
            Ystar = gp(Y, e12345_value)
            F = ip(Ystar, e12_value)
            return F

        @gajit(mask_list=[layout.grade_mask(1), layout.rotor_mask], verbose=1, ignore_cache=True)
        def example_one_gajit(X, R):
            Y = 1 + (R * X * (~R))
            D = Y * (e1 ^ e2 ^ e3 ^ einf ^ e0)
            F = D | (e1 ^ e2)
            return F


        ## ENSURE ALL THE ALGORITHMS GIVE THE SAME ANSWER
        X = layout.randomMV()(1)
        R = layout.randomRotor()
        true = example_one(X, R)
        for i in range(100):
            X = layout.randomMV()(1)
            R = layout.randomRotor()

            true = example_one(X,R)
            true_array = np.array([true.value, true.value, true.value])
            test_array = np.zeros((3,32))

            test_array[0, :] = example_one_faster(X.value, R.value)
            test_array[1, :] = example_one_faster_jit(X.value, R.value)
            test_array[2, :] = example_one_gajit(X,R).value

            for j in range(3):
                try:
                    np.testing.assert_almost_equal(test_array[j, :], true_array[j, :])
                except:
                    print('\n',flush=True)
                    print(layout.MultiVector(value=test_array[j, :]))
                    print(layout.MultiVector(value=true_array[j, :]))
                    print(layout.MultiVector(value=example_one_gajit.func(X.value, R.value)))
                    print('\n', flush=True)
                    np.testing.assert_almost_equal(test_array[j, :], true_array[j, :])

        nruns = 10000

        start_time = time.time()
        for i in range(nruns):
            example_one(X, R)
        t_normal = time.time() - start_time
        print('UNOPTIMISED: ', (10**6)*t_normal/nruns)

        start_time = time.time()
        for i in range(10000):
            example_one_faster(X.value, R.value)
        t_hand = time.time() - start_time
        print('HAND: ', (10**6)*t_hand/nruns)

        start_time = time.time()
        for i in range(10000):
            example_one_faster_jit(X.value, R.value)
        t_jit = time.time() - start_time
        print('HAND + JIT: ', (10**6)*t_jit/nruns)

        start_time = time.time()
        for i in range(10000):
            example_one_gajit(X, R)
        t_gajit = time.time() - start_time
        print('GAJIT: ', (10**6)*t_gajit/nruns)

        example_one_gajit.c_to_so()
        fso = example_one_gajit.wrap_so()

        start_time = time.time()
        for i in range(10000):
            example_one_gajit(X, R)
        t_fso = time.time() - start_time
        print('fso: ', (10**6)*t_fso/nruns)

    def test_rotor_between_lines(self):

        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)

        from clifford.tools.g3c import val_rotor_between_lines, \
            val_rotor_between_objects_root, random_line, val_normalised

        e1 = blades['e1']
        e2 = blades['e2']
        e3 = blades['e3']
        e123 = blades['e123']
        e12 = blades['e12']
        eo = stuff['eo']
        einf = stuff['einf']

        e0 = eo

        def example_rotor_lines(L, M):
            K = 2 + M * L + L * M
            A = -(K | e0) * einf
            C = 2 * (K - A)
            D = 1 - (A / C)
            R = D * (1 + M*L)
            V = R / abs(R)
            return V

        gp = layout.gmt_func
        op = layout.omt_func
        ip = layout.imt_func
        rev = layout.adjoint_func

        e12_value = (e1 ^ e2).value
        e123_value = (e1 ^ e2 ^ e3).value

        def example_rotor_lines_faster(X, R):
            return val_rotor_between_lines(X, R)

        @gajit(mask_list=[layout.grade_mask(3), layout.grade_mask(3)], verbose=1, ignore_cache=True)
        def example_rotor_lines_gajit(L, M):
            K = 2 + M * L + L * M
            A = -(K | e0) * einf
            C = 2 * (K - A)
            D = 1 - (A / C)
            R = D * (1 + M*L)
            V = R / abs(R)
            return V

        example_rotor_lines_gajit.c_to_so()
        fso = example_rotor_lines_gajit.wrap_so()

        ## ENSURE ALL THE ALGORITHMS GIVE THE SAME ANSWER
        L = random_line()
        M = random_line()
        true = example_rotor_lines(L, M)
        print(true)
        print(example_rotor_lines_gajit(L, M))
        for i in range(100):
            L = random_line()
            M = random_line()

            true = example_rotor_lines(L,M)
            true_array = np.array([true.value, true.value, true.value])
            test_array = np.zeros((3,32))

            test_array[0, :] = example_rotor_lines_faster(L.value, M.value)
            test_array[1, :] = val_rotor_between_objects_root(L.value, M.value)
            test_array[2, :] = example_rotor_lines_gajit(L,M).value

            for j in range(3):
                try:
                    np.testing.assert_almost_equal(test_array[j, :], true_array[j, :], 4)
                except:
                    print('\n',flush=True)
                    print(j)
                    print(layout.MultiVector(value=test_array[j, :]))
                    print(layout.MultiVector(value=true_array[j, :]))
                    print('\n', flush=True)
                    np.testing.assert_almost_equal(test_array[j, :], true_array[j, :], 4)


        start_time = time.time()
        for i in range(10000):
            example_rotor_lines(L, M)
        t_normal = time.time() - start_time
        print('UNOPTIMISED: ', t_normal)

        start_time = time.time()
        for i in range(10000):
            example_rotor_lines_faster(L.value, M.value)
        t_hand = time.time() - start_time
        print('OPTIMISED LINES: ', t_hand)

        start_time = time.time()
        for i in range(10000):
            val_rotor_between_objects_root(L.value, M.value)
        t_jit = time.time() - start_time
        print('OPTIMISED OBJECTS: ', t_jit)

        start_time = time.time()
        for i in range(10000):
            fso(L.value, M.value)
        t_gajit = time.time() - start_time
        print('GAJIT: ', t_gajit)




class TestCtypesWrapperExamples(unittest.TestCase):

    def setUp(self):
        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)
        self.layout = layout
        self.blades = blades

    def test_basic_map(self):

        @gajit(mask_list=[np.ones(32)], ignore_cache=True)
        def basic_map_test(A):
            C = A
            return C

        res = self.layout.randomMV()
        basic_map_test.c_to_so()
        fso = basic_map_test.wrap_so()
        res2 = fso(res.value)
        np.testing.assert_almost_equal((res).value, res2, 5)

    def test_basic_constant_addition(self):
        @gajit(mask_list=[np.ones(32)], ignore_cache=True)
        def basic_addition(A):
            C = 3.14159 + A
            return C

        res = self.layout.randomMV()
        basic_addition.c_to_so()
        fso = basic_addition.wrap_so()
        res2 = fso(res.value)

        print((3.14159 + res))
        print(self.layout.MultiVector(value=res2))

        np.testing.assert_almost_equal((3.14159 + res).value, res2, 5)

    def test_basic_mv_addition(self):
        @gajit(mask_list=[np.ones(32), self.layout.grade_mask(3)], ignore_cache=True)
        def basic_mv_addition(A, D):
            C = D + A
            return C

        A = self.layout.randomMV()
        D = self.layout.randomMV()(3)
        basic_mv_addition.c_to_so()
        fso = basic_mv_addition.wrap_so()
        res2 = fso(A.value, D.value)

        np.testing.assert_almost_equal((A + D).value, res2, 5)

    def test_basic_reverse(self):

        @gajit(mask_list=[np.ones(32)], ignore_cache=True)
        def basic_reverse(A):
            C = ~A
            return C

        res = self.layout.randomMV()
        basic_reverse.c_to_so()
        fso = basic_reverse.wrap_so()
        res2 = fso(res.value)
        np.testing.assert_almost_equal((~res).value, res2, 5)

    def test_basic_combo(self):

        e1 = self.blades['e1']
        e2 = self.blades['e2']
        e3 = self.blades['e3']

        @gajit(mask_list=[self.layout.grade_mask(1), self.layout.rotor_mask], ignore_cache=True)
        def basic_combo(A, R):
            C = 3.14 + R*A*~R
            D = C*(e1^e2^e3)
            E = D|(e1^e2)
            return E

        A = self.layout.randomMV()(1)
        R = self.layout.randomRotor()
        basic_combo.c_to_so()
        fso = basic_combo.wrap_so()

        res2 = fso(A.value, R.value)
        res = ((3.14 + R*A*~R)*(e1^e2^e3))|(e1^e2)

        np.testing.assert_almost_equal(res.value, res2, 5)


class TestBasicExamples(unittest.TestCase):

    def setUp(self):
        layout_orig, blades_orig = Cl(3)
        layout, blades, stuff = conformalize(layout_orig)
        self.layout = layout
        self.blades = blades

    def test_basic_map(self):

        @gajit(mask_list=[np.ones(32)], ignore_cache=True)
        def basic_map_test(A):
            C = A
            return C

        res = self.layout.randomMV()
        res2 = basic_map_test(res)
        np.testing.assert_almost_equal((res).value, res2.value)

    def test_basic_addition(self):

        @gajit(mask_list=[np.ones(32)], ignore_cache=True)
        def basic_addition(A):
            C = 3.14159 + A
            return C

        res = self.layout.randomMV()
        res2 = basic_addition(res)
        np.testing.assert_almost_equal((3.14159 + res).value,res2.value)

    def test_basic_reverse(self):

        @gajit(mask_list=[np.ones(32)], ignore_cache=True)
        def basic_reverse(A):
            C = ~A
            return C

        res = self.layout.randomMV()
        res2 = basic_reverse(res)
        np.testing.assert_almost_equal((~res).value, res2.value)

    # def test_basic_rotor(self):
    #
    #     @gajit(mask_list=[self.layout.grade_mask(1), self.layout.rotor_mask], ignore_cache=True)
    #     def basic_rotor(A, R):
    #         C = R*A*~R
    #         return C
    #
    #     A = self.layout.randomMV()(1)
    #     R = self.layout.randomRotor()
    #     res2 = basic_rotor(A, R)
    #     np.testing.assert_almost_equal((R*A*~R).value, res2.value)

    def test_basic_combo(self):

        e1 = self.blades['e1']
        e2 = self.blades['e2']
        e3 = self.blades['e3']

        @gajit(mask_list=[self.layout.grade_mask(1), self.layout.rotor_mask], ignore_cache=True)
        def basic_combo(A, R):
            C = 3.14 + R*A*~R
            D = C*(e1^e2^e3)
            E = D|(e1^e2)
            return E

        A = self.layout.randomMV()(1)
        R = self.layout.randomRotor()
        res2 = basic_combo(A, R)
        res = ((3.14 + R*A*~R)*(e1^e2^e3))|(e1^e2)
        np.testing.assert_almost_equal(res.value, res2.value)


class TestTimingExamples(unittest.TestCase):

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

        @gajit(mask_list=[layout.grade_mask(3), layout.grade_mask(1)], verbose=1, ignore_cache=True)
        def pp_algo_wrapped(C, P):
            Cplane = C ^ einf
            Cd = C * (Cplane)
            L = (Cplane * P * Cplane) ^ Cd ^ einf
            pp = Cd ^ (L * e12345)
            return pp

        def pp_algo(C, P):
            Cplane = C ^ einf
            Cd = C * (Cplane)
            L = (Cplane * P * Cplane) ^ Cd ^ einf
            pp = Cd ^ (L * e12345)
            return pp

        for i in range(1000):
            P = random_conformal_point()
            C = random_circle()

            np.testing.assert_almost_equal(pp_algo(C, P).value, pp_algo_wrapped(C, P).value, 5)

        start_time = time.time()
        for i in range(10000):
            pp_algo_wrapped(C, P)
        t_gaalop = time.time() - start_time
        print('GAJIT ALGO: ', t_gaalop)

        start_time = time.time()
        for i in range(10000):
            pp_algo(C, P)
        t_trad = time.time() - start_time
        print('TRADITIONAL: ', t_trad)


        pp_algo_wrapped.c_to_so()
        fso = pp_algo_wrapped.wrap_so()
        for i in range(1000):
            P = random_conformal_point()
            C = random_circle()
            np.testing.assert_almost_equal(fso(C.value, P.value), pp_algo(C, P).value, 5)

        start_time = time.time()
        for i in range(10000):
            fso(C.value, P.value)
        t_fso = time.time() - start_time
        print('fso: ', t_fso)

        print('T/G: ', t_trad / t_gaalop)
        print('T/fso: ', t_trad / t_fso)


if __name__ == '__main__':
    unittest.main()