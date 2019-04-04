from __future__ import print_function
from galgebra import ga
from sympy import *
from clifford.g3c import layout as clifford_layout
from clifford.g3c import blades as clifford_blades
import inspect


g3metric = '1 0 0 0 0, 0 1 0 0 0, 0 0 1 0 0, 0 0 0 1 0, 0 0 0 0 -1'
coords = symbols('x y z m n')
g3c = ga.Ga('e_1 e_2 e_3 e_4 e_5',g=g3metric, coords=coords)

# Give consistent names to our multivectors
e_1,e_2,e_3,e_4,e_5 = g3c.mv()
basis_array = [e_1,e_2,e_3,e_4,e_5]

# This allows us to use normal clifford names with galgebra
for k,v in clifford_blades.items():
    running = 1
    for i in k[1:]:
        running = running^basis_array[int(i)-1]
    exec(k + '='+ str(running))
blade_names = clifford_layout.names
blade_names[0] = '1'

# Define our null vectors
ninf = e4 + e5
einf = ninf
no = 0.5*(e4 - e5)
eo = -no
e0 = eo

# Define an up mapping
def up(x):
    return x + 0.5*x*x*ninf - no

# Define a homo function as per clifford
def homo(x):
    y = (-x|ninf)
    adj = (~y)
    madj = adj*y
    return x*(adj/madj)

# Define a down mapping
E0 = ninf^(-no)
def down(x):
    return (homo(x)^E0)*E0

# Define a normalisation
def normalised(x):
    return x/(sqrt(g3c.scalar_part(x*~x)))

def normalise_n_minus_1(x):
    return -x/g3c.scalar_part(x|ninf)


# Define some useful things
I5 = e12345
I3 = e123

# This is a full multivector symbol generator
def generate_new_mv(mv_name, blade_mask=None):
    coefs = symbols(mv_name+'_00:%d'%clifford_layout.gaDims)
    running = 0
    if blade_mask is None:
        for i,c in enumerate(coefs):
            running += c*eval(blade_names[i])
        return running, coefs
    else:
        mv_symbs = []
        for i,c in enumerate(coefs):
            if blade_mask[i]:
                running += c*eval(blade_names[i])
                mv_symbs.append(c)
        return running, mv_symbs

# This is a euclidean point
def generate_new_euc_point(mv_name):
    coefs = symbols(mv_name+'_00:%d'%clifford_layout.gaDims)
    running = 0
    mv_symbs = []
    for i,c in enumerate(coefs):
        if i in range(1,4):
            running += c*eval(blade_names[i])
            mv_symbs.append(c)
    return running, mv_symbs

# This is a euclidean point
def generate_new_conformal_point(mv_name):
    running,mv_symbs = generate_new_euc_point(mv_name)
    return up(running), mv_symbs


def unindent(source):
    wrapfree = source[source.find('\n')+1:]
    point = wrapfree.find('def ')
    return '\n'.join([line[point:] for line in iter(wrapfree.splitlines())])


def galgebra_function(galgebra_function, inputs, outputs, gms, function_name):
    # Reevaluate the function in the current context to make sure sympy handles it all
    exec(unindent(galgebra_function))
    print(unindent(galgebra_function))
    print( locals())
    newf = locals()[function_name]

    inp_objs = []
    for inp, grade_mask in zip(inputs, gms):
        X, X_symbs = generate_new_mv(inp, grade_mask)
        inp_objs.append(X)

    func_string = 'def '+function_name+'('
    for i, inp in enumerate(inputs):
        if i == 0:
            func_string += inp
        else:
            func_string += ',' + inp
    func_string += '):\n'
    for op in outputs:
        func_string += '    ' + op + ' = np.zeros(32)\n'

    input_remapping = ''
    for inp_it, inp in enumerate(inputs):
        for k in range(32):
            if gms[inp_it][k]:
                input_remapping += '    ' + inp + '_' + str(k) + '=' + inp + '[' + str(k) + ']\n'
    func_string += input_remapping

    function_result = newf(*inp_objs)
    opcoefs = function_result.blade_coefs()
    for i in range(32):
        if opcoefs[i] != 0:
            func_string += '    ' + outputs[0] + '[' + str(i) + ']=' + str(opcoefs[i]) + '\n'
    func_string += '    return ' + outputs[0]
    func_string = '@numba.njit\n' + func_string
    return func_string
