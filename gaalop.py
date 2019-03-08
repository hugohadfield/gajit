
import os
os.environ['NUMBA_DISABLE_PARALLEL'] = '1'

USE_NUMBA = True

if USE_NUMBA == False:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

from clifford.tools.g3c import *
from clifford.g3c import *
import numpy as np
import re
import time
import numba







def activate_gaalop(gaalop_script):
    """
    This function should take the CLUSCRIPT and pass it to gaalop over cli
    It should then activate the gaalop optimisation and return the c code result
    as a string
    """
    return activate_gaalop_manual_via_files(gaalop_script)



def activate_gaalop_manual_via_files(gaalop_script):
    """
    This initiates the manual gaalop process via files
    """
    # Write the gaalop script to the local directory
    with open('gaalop_script.clu', 'w') as fobj:
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
    with open('gaalop_script.c', 'r') as fobj:
        c_code_result = fobj.read()
    print('\n')
    print('gaalop_script.c FOUND, CONTINUING')
    print('\n')
    return c_code_result


def activate_gaalop_manual(gaalop_script):
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


def define_gaalop_function(inputs=[],blade_mask_list=None,outputs=[],body=''):
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


def process_output_body(output_text,inputs=[],outputs=[],function_name='gaalop'):
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
        for j in range(32):
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
        body=None,
        function_name='gaalop_func'):

        self.inputs = inputs
        self.blade_mask_list = blade_mask_list
        self.outputs = outputs
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
            body=self.body)
        return self.gaalop_input

    def activate_gaalop(self):
        """
        Calls gaalop with the CLUScript specified
        """
        self.gaalop_output = activate_gaalop(self.gaalop_input)
        return self.gaalop_output

    def process_gaalop_output(self):
        """
        Processes the c output of gaalop to make it into JITable python
        """
        self.python_text = process_output_body(self.gaalop_output,
            inputs=self.inputs,
            outputs=self.outputs,
            function_name=self.function_name)
        return self.python_text

    def process_python_function(self):
        """
        Evaluates the python text to generate a function in scope
        """
        exec(self.python_text)
        self.func = locals()[self.function_name]
        return self.func

    def compile(self, verbose=False):
        """ 
        Runs the full compilation pipeline 
        """
        if self.check_compilable():
            if not verbose:
                self.define_gaalop_function()
                self.activate_gaalop()
                self.process_gaalop_output()
                self.process_python_function()
                self._compiled = True
            else:
                print( self.define_gaalop_function() )
                print( self.activate_gaalop() )
                print( self.process_gaalop_output() )
                print( self.process_python_function() )
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
                return [layout.MultiVector(value=a) for a in self.func(*args)]
            else:
                return self.func(*args)
        else:
            self.compile()
            return self.__call__(*args)






























if __name__ == '__main__':

    # These are a set of masks we can apply to the input multivectors 
    # so we dont have to define as many symbols
    onevectormask = np.abs((layout.randomMV()(1).value)) > 0
    twovectormask = np.abs((layout.randomMV()(2).value)) > 0
    threevectormask = np.abs((layout.randomMV()(3).value)) > 0
    fourvectormask = np.abs(((e1+e2+e3+e4+e5)*e12345).value) > 0
    fivevectormask = np.abs((e12345).value) > 0


    def traditional_algo():
        L = (P|S)
        return L*einf*L

    @numba.njit
    def traditional_algo_fast(P_val,S_val):
        L = imt_func(P_val,S_val)
        return gmt_func(gmt_func(L,ninf_val),L)
    
    test_algo = Algorithm(
        inputs=['P','S'],
        blade_mask_list=[onevectormask,fourvectormask],
        outputs=['C'],
        body="""
        C = (P.S)*einf*(P.S);
        """)

    P = random_conformal_point()
    S = random_sphere()

    res_val = test_algo(P.value, S.value)
    C = layout.MultiVector(value=res_val)
    print( C )

    


    # Ensure that the f
    print( traditional_algo() )
    print( layout.MultiVector(value=traditional_algo_fast(P.value, S.value) ))

    start_time = time.time()
    for i in range(10000):
        test_algo.func(P.value, S.value)
    t_gaalop = time.time() - start_time
    print('GAALOP ALGO: ', t_gaalop)

    start_time = time.time()
    for i in range(10000):
        traditional_algo_fast(P.value, S.value)
    t_trad = time.time() - start_time
    print('TRADITIONAL: ', t_trad)

    print('T/G: ', t_trad/t_gaalop)

