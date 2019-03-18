
from clifford.g3c import *
import os
import numpy as np
import re
import time
import numba

from pathlib import Path

import subprocess
import inspect

from enum import Enum


# These are a set of masks we can apply to the input multivectors
# so we dont have to define as many symbols
class GradeMasks(Enum):
    onevectormask = tuple( np.abs((layout.randomMV()(1).value)) > 0 )
    twovectormask = tuple( np.abs((layout.randomMV()(2).value)) > 0 )
    threevectormask = tuple( np.abs((layout.randomMV()(3).value)) > 0)
    fourvectormask = tuple( np.abs(((e1+e2+e3+e4+e5)*e12345).value) > 0 )
    fivevectormask = tuple( np.abs((e12345).value) > 0 )


# USEFUL DEFINITIONS
e0 = eo
GAALOP_CLI_HOME = os.environ['GAALOP_CLI_HOME']
GAALOP_ALGEBRA_HOME = os.environ['GAALOP_ALGEBRA_HOME']
symbols_py2g = {'|': '.',
                'e12345': '(e1^e2^e3^einf^e0)',
                'e123': '(e1^e2^e3)'
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
    function_def = cleaned_script.split(':', 1)[0]
    cleaned_script = cleaned_script.split(':', 1)[1]
    function_def = function_def.split('def', 1)[1]
    function_name = function_def.split('(', 1)[0].strip()
    argument_string = function_def.split('(', 1)[1].split(')', 1)[0]
    inputs = [a.strip() for a in argument_string.split(',')]

    # Run the conversion
    gaalop_script = cleaned_script.strip()
    for k, v in symbols_py2g.items():
        gaalop_script = gaalop_script.replace(k, v)

    # Assert that all lines are terminated with semi colons
    content = []
    for line in gaalop_script.split('\n'):
        l = line.strip()
        if l[-1] != ';':
            content.append(l + ';')
        else:
            content.append(l)
    gaalop_script = '\n'.join(content)

    # Extract the intermediates
    intermediates = []
    for line in cleaned_script.split('\n'):
        if len(line.strip()) > 0:
            sym = line.split('=', 1)[0].strip()
            if sym not in inputs and sym not in outputs:
                intermediates.append(sym)

    return gaalop_script, inputs, outputs, intermediates, function_name


def py2Algorithm(python_function, mask_list=None):
    gaalop_script, inputs, outputs, intermediates, function_name = py2gaalop(python_function)
    return Algorithm(
        body=gaalop_script,
        inputs=inputs,
        outputs=outputs,
        blade_mask_list=mask_list,
        function_name=function_name,
        intermediates=intermediates)


def gajit(mask_list=None, verbose=0):
    def gaalop_wrapper(func):
        python_function = inspect.getsource(func)
        algo = py2Algorithm(python_function, mask_list=mask_list)
        algo.compile(verbose=verbose)
        return algo

    return gaalop_wrapper



def activate_gaalop(gaalop_script,*args):
    """
    This function should take the CLUSCRIPT and pass it to gaalop over cli
    It should then activate the gaalop optimisation and return the c code result
    as a string
    """
    return activate_gaalop_CLI(gaalop_script,*args)


def activate_gaalop_CLI(gaalop_script, name, *args):
    """
    This function should take the CLUSCRIPT and pass it to gaalop over cli
    It should then activate the gaalop optimisation and return the c code result
    as a string
    """
    fname = name.strip()
    wd = os.getcwd()
    # Write the gaalop script to the local directory
    with open(fname+'.clu', 'w') as fobj:
        print(gaalop_script,file=fobj)
    # Call the Gaalop CLI
    os.chdir(GAALOP_CLI_HOME)
    if os.name == 'nt':
        subprocess.run(['java','-jar', 'starter-1.0.0.jar',
            '-algebraName','5d',
            '-o', wd, 
            '-optimizer','de.gaalop.tba.Plugin',
            '-generator','de.gaalop.cpp.Plugin',
            '-algebraBaseDir',GAALOP_ALGEBRA_HOME,
            '-i','/../../../../../../../../../../../../../../'+wd[3:]+'/'+fname+'.clu'
            ])
    else:
        subprocess.run(['java','-jar', 'starter-1.0.0.jar',
            '-algebraName','5d',
            '-o', wd, 
            '-optimizer','de.gaalop.tba.Plugin',
            '-generator','de.gaalop.cpp.Plugin',
            '-algebraBaseDir',GAALOP_ALGEBRA_HOME,
            '-i','/../../../../../../../../../../../../../../'+wd+'/'+fname+'.clu'
            ])
    os.chdir(wd)
    # Load the file
    with open(fname+'.c', 'r') as fobj:
        c_code_result = fobj.read()
    return c_code_result


def activate_gaalop_manual_via_files(gaalop_script, name, *args):
    """
    This initiates the manual gaalop process via files
    """
    fname = name.strip()
    # Write the gaalop script to the local directory
    with open(fname+'.clu', 'w') as fobj:
        print(gaalop_script,file=fobj)
    print('\n\n')
    print('OPEN GAALOP AND LOAD THE FILE gaalop_script.clu FROM THIS DIRECTORY')
    print('OPTIMISE THE CODE AND SAVE THE C CODE IN THiS DIRECTORY AS gaalop_script.c')
    print('Press Ctrl-D (linux/mac) or Ctrl-Z ( windows ) and hit enter to continue once this is done.')
    print('\n')
    while True:
        try:
            line = input()
        except EOFError:
            break
    # Load the file
    with open(fname+'.c', 'r') as fobj:
        c_code_result = fobj.read()
    print('\n')
    print(fname+'.c FOUND, CONTINUING')
    print('\n')
    return c_code_result


def activate_gaalop_manual(gaalop_script, *args):
    """ 
    This initiates the manual gaalop process
    """
    print('\n\n')
    print('COPY AND PASTE THE FOLLOWING CODE INTO GAALOP.')
    print('\n')
    print(gaalop_script)
    print('\n')
    print('COMPILE THE CODE AND COPY AND PASTE THE RESULTANT C CODE HERE')
    print('After pasting press Ctrl-D (linux/mac) or Ctrl-Z ( windows ) and hit enter to continue.')
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    c_code_result = '\n'.join(contents)
    print('\n')
    print('CODE RECEIVED, CONTINUING')
    print('\n')
    return c_code_result















gaalop_list = [1+0*e1,
e1,
e2,
e3,
einf,
eo,
e1^e2,
e1^e3,
e1^einf,
e1^eo,
e2^e3,
e2^einf,
e2^eo,
e3^einf,
e3^eo,
einf^eo,
e1^e2^e3,
e1^e2^einf,
e1^e2^eo,
e1^e3^einf,
e1^e3^eo,
e1^einf^eo,
e2^e3^einf,
e2^e3^eo,
e2^einf^eo,
e3^einf^eo,
e1^e2^e3^einf,
e1^e2^e3^eo,
e1^e2^einf^eo,
e1^e3^einf^eo,
e2^e3^einf^eo,
e1^e2^e3^e4^e5]

gaalop_map = np.array([x.value for x in gaalop_list]).T
inverse_gaalop_map = np.linalg.inv(gaalop_map)


def convert_gaalop_vector(gv):
    """
    This function takes an array of coefficients defined in the gaalop standard ordering
    and converts it to the clifford standard ordering multivector object
    """
    return layout.MultiVector(value=gaalop_map@gv)


def convert_clifford_to_gaalop(mv):
    """
    This function takes a MultiVector and converts it into an array in the GAALOP
    standard ordering
    """
    return inverse_gaalop_map@mv.value


def map_multivector(mv_name='X',blade_mask=np.ones(32)):
    mapping_function = ''
    blade_names = ['1'] + list(layout.blades.keys())
    first_flag = True
    for i in range(32):
        if blade_mask[i] == 1:
            blade_name = blade_names[i]
            final_blade_name = ''
            if blade_name == 'e4':
                final_blade_name = '(0.5*einf - e0)'
            elif blade_name == 'e5':
                final_blade_name = '(0.5*einf + e0)'
            else:
                if len(blade_name) > 1:
                    for j,l in enumerate(blade_name):
                        yielded_string = l
                        if blade_name[1] == '4' and j==0:
                            yielded_string = '(0.5*einf - e0)'
                        if blade_name[1] == '4' and j==1:
                            yielded_string = ''
                        if j > 1:
                            if blade_name[j] != 'e':
                                if blade_name[j] == '4':
                                    yielded_string = '^(0.5*einf - e0)'
                                elif blade_name[j] == '5':
                                    yielded_string = '^(0.5*einf + e0)'
                                else:
                                    yielded_string = '^e'+ blade_name[j]
                        final_blade_name+=yielded_string
                else:
                    final_blade_name = blade_name
            param_i = mv_name+'_'+str(i)
            if first_flag:
                mapping_function += param_i + '*' + final_blade_name
                first_flag = False
            else:
                mapping_function += ' + ' + param_i + '*' + final_blade_name
    return mv_name+' = ' +mapping_function+';'+'\n'


def define_gaalop_function(inputs=[],blade_mask_list=None,outputs=[],intermediates=[],body=''):
    total_string = ''
    input_string = ''
    for i,input_name in enumerate(inputs):
        if blade_mask_list is None:
            blade_mask = np.ones(32)
        else:
            blade_mask = blade_mask_list[i]
        input_string+= map_multivector(mv_name=input_name,blade_mask=blade_mask)
    total_string += input_string
    total_string += body
    output_string  = ''
    for o in outputs:
        output_string += '?'+o+';'+'\n'
    total_string += output_string
    return total_string


def process_output_body(output_text,inputs=[],outputs=[],intermediates=[],function_name='gaalop'):
    # Remove tabs
    final_text =  re.sub(r"[\t]","",output_text)
    # Try and find the first curly brakcet
    if '{' in final_text.split('\n', 1)[0]:
        final_text = final_text.split('\n',1)[1];
    # Remove the final } if it is there
    final_text =  re.sub('}',"",final_text)
    # Reformat comments
    final_text = final_text.replace('//', '#')
    # Start the advanced processing
    for input_name in inputs:
        pattern = input_name+'['
        temp_val = input_name+'temp'
        final_text = temp_val + '= np.zeros(32)\n' + final_text
        final_text = final_text.replace(pattern, temp_val+'[')
        for j in range(33):
            pattern = input_name+'_'+str(32-j)
            replacement = input_name+'['+str(32-j)+']'
            final_text = final_text.replace(pattern, replacement)
    final_text = final_text + '\nreturn '
    n = 0
    for o in outputs:
        final_text = o + '= np.zeros(32)\n' + final_text
        if n !=0:
            final_text = final_text + ',gaalop_map@'+ o + ' '
        else:
            final_text = final_text + 'gaalop_map@'+o + ' '
        n = n + 1
    for o in intermediates:
        final_text = o + '= np.zeros(32)\n' + final_text
    final_signature = 'def '+ function_name+'('
    for i,input_name in enumerate(inputs):
        if i == 0:
            final_signature += input_name
        else:
            final_signature += ',' + input_name
    final_signature = final_signature + '):\n'
    final_text = final_signature + final_text
    final_text = final_text.replace('\n', '\n    ')
    final_text = '@numba.njit\n' + final_text
    return final_text

















class Algorithm:
    """
    This class provides the ability to interface easily with gaalop

    """
    def __init__(self, 
        inputs=[], 
        blade_mask_list=[], 
        outputs=[], 
        intermediates=[],
        body=None,
        function_name='gaalop_func'):

        self.inputs = inputs
        self.blade_mask_list = blade_mask_list
        self.outputs = outputs
        self.intermediates = intermediates
        self.body = body
        self.function_name = function_name

        self.gaalop_input = None
        self._compiled = False

    def check_compilable(self):
        # TODO add some actual checking here to see 
        # if we have specified enough stuff
        return True

    def define_gaalop_function(self):
        """
        Generates the proper CLUScript code from the input
        """
        self.gaalop_input = define_gaalop_function(inputs=self.inputs, 
            blade_mask_list=self.blade_mask_list, 
            outputs=self.outputs, 
            intermediates=self.intermediates,
            body=self.body)
        return self.gaalop_input

    def activate_gaalop(self):
        """
        Calls gaalop with the CLUScript specified
        """
        self.gaalop_output = activate_gaalop(self.gaalop_input, self.function_name)
        return self.gaalop_output

    def process_gaalop_output(self):
        """
        Processes the c output of gaalop to make it into JITable python
        """
        self.python_text = process_output_body(self.gaalop_output,
            inputs=self.inputs,
            outputs=self.outputs,
            intermediates=self.intermediates,
            function_name=self.function_name)
        return self.python_text

    def process_python_function(self):
        """
        Evaluates the python text to generate a function in scope
        """
        exec(self.python_text)
        self.func = locals()[self.function_name]
        return self.func

    def compile(self, verbose=0):
        """ 
        Runs the full compilation pipeline 
        """
        if self.check_compilable():
            if verbose >= 1:
                print('Running gaalop function definition')
            s0 = self.define_gaalop_function()
            if verbose >= 2:
                print(s0)

            if verbose >= 1:
                print('Activating gaalop')
            s1 = self.activate_gaalop()
            if verbose >= 2:
                print(s1)

            if verbose >= 1:
                print('Processing gaalop output')
            s2 = self.process_gaalop_output()
            if verbose >= 2:
                print(s2)

            if verbose >= 1:
                print('Loading python function')
            s3 = self.process_python_function()
            if verbose >= 2:
                print(s3)

            if verbose >= 1:
                print('Complete')
            self._compiled = True
        else:
            raise ValueError("""The function is not sufficiently defined.
                Please check that you have specified at least one output and a function body.
                """)

    def __call__(self, *args):
        """ 
        Allows us to use the instance as a function
        If it has not compiled everything, it does it now
        """
        if self._compiled:
            if len(self.outputs) > 1:
                return [layout.MultiVector(value=a) for a in self.func(*[a.value for a in args])]
            else:
                return layout.MultiVector(value=self.func(*[a.value for a in args]))
        else:
            self.compile()
            return self.__call__(*args)


