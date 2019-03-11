
from gaalop import Algorithm, GradeMasks
import inspect


symbols_py2g = {'|':'.',
                'e12345':'(e1^e2^e3^einf^e0)',
                'e123':'(e1^e2^e3)'
                }
symbols_g2py = {v: k for k, v in symbols_py2g.items()}


def py2gaalop(python_script):
    """ 
    This transpiles a python function to a gaalop compatible CLUSCRIPT
    This supports a very minimal subset of python, no flow control etc
    """
    # Get rid of front and back whitespace
    cleaned_script = python_script.strip()

    # Extract the ouputs
    split_function = cleaned_script.split('return')
    cleaned_script = split_function[0]
    output_string = split_function[1]
    outputs = [a.strip() for a in output_string.split(',')]
    
    # Extract the name and inputs
    function_def = cleaned_script.split(':',1)[0]
    cleaned_script = cleaned_script.split(':',1)[1]
    function_def = function_def.split('def',1)[1]
    function_name = function_def.split('(',1)[0].strip()
    argument_string = function_def.split('(',1)[1].split(')',1)[0]
    inputs = [a.strip() for a in argument_string.split(',')]

    # Run the conversion
    gaalop_script = cleaned_script.strip()
    for k,v in symbols_py2g.items():
        gaalop_script = gaalop_script.replace(k,v)

    # Assert that all lines are terminated with semi colons
    content = []
    for line in gaalop_script.split('\n'):
        l = line.strip()
        if l[-1] != ';':
            content.append(l+';')
        else:
            content.append(l)
    gaalop_script = '\n'.join(content)

    # Extract the intermediates
    intermediates = [ ]
    for line in cleaned_script.split('\n'):
        if len(line.strip()) > 0:
            sym = line.split('=',1)[0].strip()
            if sym not in inputs and sym not in outputs:
                intermediates.append(sym)

    return gaalop_script, inputs, outputs, intermediates, function_name


def py2Algorithm(python_function,mask_list=None):
    gaalop_script, inputs, outputs, intermediates, function_name = py2gaalop(python_function)
    return Algorithm(
        body=gaalop_script,
        inputs=inputs,
        outputs=outputs,
        blade_mask_list=mask_list,
        function_name=function_name,
        intermediates=intermediates)


def gaalop(mask_list=None, verbose=0):
    def gaalop_wrapper(func):
        python_function = inspect.getsource(func)
        algo = py2Algorithm(python_function, mask_list=mask_list)
        algo.compile(verbose=verbose)
        return algo
    return gaalop_wrapper

@gaalop(mask_list=[GradeMasks.onevectormask.value,GradeMasks.threevectormask.value],verbose=1)
def pp_algo_wrapped(P, C):
    Cplane = C^einf
    Cd = C*(Cplane)
    L = (Cplane*P*Cplane)^Cd^einf
    pp = Cd^(L*e12345)
    return pp

def pp_algo(P, C):
    Cplane = C^einf
    Cd = C*(Cplane)
    L = (Cplane*P*Cplane)^Cd^einf
    pp = Cd^(L*e12345)
    return pp




from clifford.tools.g3c import *
from clifford.g3c import *
import time

P = random_conformal_point()
C = random_circle()

print(pp_algo(P,C))
print(pp_algo_wrapped(P,C))

start_time = time.time()
for i in range(10000):
    pp_algo_wrapped(P,C)
t_gaalop = time.time() - start_time
print('GAALOP ALGO 2: ', t_gaalop)

start_time = time.time()
for i in range(10000):
    pp_algo(P,C)
t_trad = time.time() - start_time
print('TRADITIONAL 2: ', t_trad)

print('T/G: ', t_trad/t_gaalop)