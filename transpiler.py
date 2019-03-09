
from gaalop import Algorithm

symbols_py2g = {'|':'.'}
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
    gaalop_script = cleaned_script

    # Extract the intermediates
    intermediates = [ ]
    for line in cleaned_script.split('\n'):
        if len(line.strip()) > 0:
            sym = line.split('=',1)[0].strip()
            if sym not in inputs and sym not in outputs:
                intermediates.append(sym)

    return gaalop_script, inputs, outputs, intermediates, function_name


test_string ="""
def test_func(A,B,C):
    D = A + B
    T = (B|C)
    E = T^einf
    F = A*((C^e0)^einf)
    return D,E,F
"""
gaalop_script, inputs, outputs, intermediates, function_name = py2gaalop(test_string)
print(gaalop_script)
print(inputs)
print(outputs)
print(intermediates)
print(function_name)
