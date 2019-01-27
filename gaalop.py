
USE_NUMBA = True
NTESTS = 10000000

if USE_NUMBA == False:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

from clifford.g3c import *
import numpy as np
import re

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
    final_text =  re.sub(r"[\t]","",output_text)
    final_text = final_text.replace('//', '#')
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
    return final_text

if __name__ == '__main__':

    import numba

    fourvectormask = np.abs(((e1+e2+e3+e4+e5)*e12345).value) > 0
    twovectormask = np.abs((layout.randomMV()(2).value)) > 0
    threevectormask = np.abs((layout.randomMV()(3).value)) > 0
    body="""
    C=P*L*P;
    """
    print(define_gaalop_function(inputs=['P','L'], blade_mask_list=[fourvectormask,threevectormask], outputs=['C'], body=body ))

    output_text="""
    P[26] = P_26 / 2.0 + P_27 / 2.0; // e1 ^ (e2 ^ (e3 ^ einf))
    P[27] = (-P_26) + P_27; // e1 ^ (e2 ^ (e3 ^ e0))
    L[17] = L_17 / 2.0 + L_18 / 2.0; // e1 ^ (e2 ^ einf)
    L[18] = (-L_17) + L_18; // e1 ^ (e2 ^ e0)
    L[19] = L_19 / 2.0 + L_20 / 2.0; // e1 ^ (e3 ^ einf)
    L[20] = (-L_19) + L_20; // e1 ^ (e3 ^ e0)
    L[22] = L_22 / 2.0 + L_23 / 2.0; // e2 ^ (e3 ^ einf)
    L[23] = (-L_22) + L_23; // e2 ^ (e3 ^ e0)
    C[1] = (-(((-(P[26] * L_21)) + P_28 * L[19] + (-(P_29 * L[17]))) * P[27])) + (-((P[27] * L_21 + (-(P_28 * L[20])) + P_29 * L[18]) * P[26])) + (-((P[26] * L[20] + (-(P[27] * L[19])) + P_29 * L_16) * P_28)) + (-(((-(P[26] * L[18])) + P[27] * L[17] + (-(P_28 * L_16))) * P_29)); // e1
    C[2] = (P[26] * L_24 + (-(P_28 * L[22])) + P_30 * L[17]) * P[27] + ((-(P[27] * L_24)) + P_28 * L[23] + (-(P_30 * L[18]))) * P[26] + ((-(P[26] * L[23])) + P[27] * L[22] + (-(P_30 * L_16))) * P_28 + (-(((-(P[26] * L[18])) + P[27] * L[17] + (-(P_28 * L_16))) * P_30)); // e2
    C[3] = (-(((-(P[26] * L_25)) + P_29 * L[22] + (-(P_30 * L[19]))) * P[27])) + (-((P[27] * L_25 + (-(P_29 * L[23])) + P_30 * L[20]) * P[26])) + ((-(P[26] * L[23])) + P[27] * L[22] + (-(P_30 * L_16))) * P_29 + (P[26] * L[20] + (-(P[27] * L[19])) + P_29 * L_16) * P_30; // e3
    C[4] = (-((P_28 * L_25 + (-(P_29 * L_24)) + P_30 * L_21) * P[26])) + (-(((-(P[26] * L_25)) + P_29 * L[22] + (-(P_30 * L[19]))) * P_28)) + (-((P[26] * L_24 + (-(P_28 * L[22])) + P_30 * L[17]) * P_29)) + (-(((-(P[26] * L_21)) + P_28 * L[19] + (-(P_29 * L[17]))) * P_30)); // einf
    C[5] = (-((P_28 * L_25 + (-(P_29 * L_24)) + P_30 * L_21) * P[27])) + (P[27] * L_25 + (-(P_29 * L[23])) + P_30 * L[20]) * P_28 + ((-(P[27] * L_24)) + P_28 * L[23] + (-(P_30 * L[18]))) * P_29 + (P[27] * L_21 + (-(P_28 * L[20])) + P_29 * L[18]) * P_30; // e0
    C[16] = (P[26] * L_16 + P_28 * L[17] + P_29 * L[19] + P_30 * L[22]) * P[27] + (P[27] * L_16 + (-(P_28 * L[18])) + (-(P_29 * L[20])) + (-(P_30 * L[23]))) * P[26] + ((-(P[26] * L[23])) + P[27] * L[22] + (-(P_30 * L_16))) * P_30 + (-((P[26] * L[20] + (-(P[27] * L[19])) + P_29 * L_16) * P_29)) + ((-(P[26] * L[18])) + P[27] * L[17] + (-(P_28 * L_16))) * P_28; // e1 ^ (e2 ^ e3)
    C[17] = (P[26] * L[18] + P[27] * L[17] + (-(P_29 * L_21)) + (-(P_30 * L_24))) * P[26] + (P[26] * L_16 + P_28 * L[17] + P_29 * L[19] + P_30 * L[22]) * P_28 + (-((P[26] * L_24 + (-(P_28 * L[22])) + P_30 * L[17]) * P_30)) + ((-(P[26] * L_21)) + P_28 * L[19] + (-(P_29 * L[17]))) * P_29 + (-(((-(P[26] * L[18])) + P[27] * L[17] + (-(P_28 * L_16))) * P[26])); // e1 ^ (e2 ^ einf)
    C[18] = (P[26] * L[18] + P[27] * L[17] + (-(P_29 * L_21)) + (-(P_30 * L_24))) * P[27] + (-((P[27] * L_16 + (-(P_28 * L[18])) + (-(P_29 * L[20])) + (-(P_30 * L[23]))) * P_28)) + ((-(P[27] * L_24)) + P_28 * L[23] + (-(P_30 * L[18]))) * P_30 + (-((P[27] * L_21 + (-(P_28 * L[20])) + P_29 * L[18]) * P_29)) + ((-(P[26] * L[18])) + P[27] * L[17] + (-(P_28 * L_16))) * P[27]; // e1 ^ (e2 ^ e0)
    C[19] = (-(((-(P[26] * L[20])) + (-(P[27] * L[19])) + (-(P_28 * L_21)) + P_30 * L_25) * P[26])) + (P[26] * L_16 + P_28 * L[17] + P_29 * L[19] + P_30 * L[22]) * P_29 + ((-(P[26] * L_25)) + P_29 * L[22] + (-(P_30 * L[19]))) * P_30 + (-(((-(P[26] * L_21)) + P_28 * L[19] + (-(P_29 * L[17]))) * P_28)) + (P[26] * L[20] + (-(P[27] * L[19])) + P_29 * L_16) * P[26]; // e1 ^ (e3 ^ einf)
    C[20] = (-(((-(P[26] * L[20])) + (-(P[27] * L[19])) + (-(P_28 * L_21)) + P_30 * L_25) * P[27])) + (-((P[27] * L_16 + (-(P_28 * L[18])) + (-(P_29 * L[20])) + (-(P_30 * L[23]))) * P_29)) + (-((P[27] * L_25 + (-(P_29 * L[23])) + P_30 * L[20]) * P_30)) + (P[27] * L_21 + (-(P_28 * L[20])) + P_29 * L[18]) * P_28 + (-((P[26] * L[20] + (-(P[27] * L[19])) + P_29 * L_16) * P[27])); // e1 ^ (e3 ^ e0)
    C[21] = (-(((-(P[26] * L[20])) + (-(P[27] * L[19])) + (-(P_28 * L_21)) + P_30 * L_25) * P_28)) + (-((P[26] * L[18] + P[27] * L[17] + (-(P_29 * L_21)) + (-(P_30 * L_24))) * P_29)) + (-((P_28 * L_25 + (-(P_29 * L_24)) + P_30 * L_21) * P_30)) + ((-(P[26] * L_21)) + P_28 * L[19] + (-(P_29 * L[17]))) * P[27] + (-((P[27] * L_21 + (-(P_28 * L[20])) + P_29 * L[18]) * P[26])); // e1 ^ (einf ^ e0)
    C[22] = (P[26] * L[23] + P[27] * L[22] + P_28 * L_24 + P_29 * L_25) * P[26] + (P[26] * L_16 + P_28 * L[17] + P_29 * L[19] + P_30 * L[22]) * P_30 + (-(((-(P[26] * L_25)) + P_29 * L[22] + (-(P_30 * L[19]))) * P_29)) + (P[26] * L_24 + (-(P_28 * L[22])) + P_30 * L[17]) * P_28 + (-(((-(P[26] * L[23])) + P[27] * L[22] + (-(P_30 * L_16))) * P[26])); // e2 ^ (e3 ^ einf)
    C[23] = (P[26] * L[23] + P[27] * L[22] + P_28 * L_24 + P_29 * L_25) * P[27] + (-((P[27] * L_16 + (-(P_28 * L[18])) + (-(P_29 * L[20])) + (-(P_30 * L[23]))) * P_30)) + (P[27] * L_25 + (-(P_29 * L[23])) + P_30 * L[20]) * P_29 + (-(((-(P[27] * L_24)) + P_28 * L[23] + (-(P_30 * L[18]))) * P_28)) + ((-(P[26] * L[23])) + P[27] * L[22] + (-(P_30 * L_16))) * P[27]; // e2 ^ (e3 ^ e0)
    C[24] = (P[26] * L[23] + P[27] * L[22] + P_28 * L_24 + P_29 * L_25) * P_28 + (-((P[26] * L[18] + P[27] * L[17] + (-(P_29 * L_21)) + (-(P_30 * L_24))) * P_30)) + (P_28 * L_25 + (-(P_29 * L_24)) + P_30 * L_21) * P_29 + (-((P[26] * L_24 + (-(P_28 * L[22])) + P_30 * L[17]) * P[27])) + ((-(P[27] * L_24)) + P_28 * L[23] + (-(P_30 * L[18]))) * P[26]; // e2 ^ (einf ^ e0)
    C[25] = (P[26] * L[23] + P[27] * L[22] + P_28 * L_24 + P_29 * L_25) * P_29 + ((-(P[26] * L[20])) + (-(P[27] * L[19])) + (-(P_28 * L_21)) + P_30 * L_25) * P_30 + (-((P_28 * L_25 + (-(P_29 * L_24)) + P_30 * L_21) * P_28)) + ((-(P[26] * L_25)) + P_29 * L[22] + (-(P_30 * L[19]))) * P[27] + (-((P[27] * L_25 + (-(P_29 * L[23])) + P_30 * L[20]) * P[26])); // e3 ^ (einf ^ e0)
    C[31] = (P[26] * L[23] + P[27] * L[22] + P_28 * L_24 + P_29 * L_25) * P_30 + (-(((-(P[26] * L[20])) + (-(P[27] * L[19])) + (-(P_28 * L_21)) + P_30 * L_25) * P_29)) + (P[26] * L[18] + P[27] * L[17] + (-(P_29 * L_21)) + (-(P_30 * L_24))) * P_28 + (-((P[26] * L_16 + P_28 * L[17] + P_29 * L[19] + P_30 * L[22]) * P[27])) + (P[27] * L_16 + (-(P_28 * L[18])) + (-(P_29 * L[20])) + (-(P_30 * L[23]))) * P[26]; // e1 ^ (e2 ^ (e3 ^ (einf ^ e0)))"""
    print(process_output_body(output_text,inputs=['P','L'],outputs=['C']))

    @numba.njit
    def gaalop(P,L):
        C= np.zeros(32)
        Ltemp= np.zeros(32)
        Ptemp= np.zeros(32)

        Ptemp[26] = P[26] / 2.0 + P[27] / 2.0; # e1 ^ (e2 ^ (e3 ^ einf))
        Ptemp[27] = (-P[26]) + P[27]; # e1 ^ (e2 ^ (e3 ^ e0))
        Ltemp[17] = L[17] / 2.0 + L[18] / 2.0; # e1 ^ (e2 ^ einf)
        Ltemp[18] = (-L[17]) + L[18]; # e1 ^ (e2 ^ e0)
        Ltemp[19] = L[19] / 2.0 + L[20] / 2.0; # e1 ^ (e3 ^ einf)
        Ltemp[20] = (-L[19]) + L[20]; # e1 ^ (e3 ^ e0)
        Ltemp[22] = L[22] / 2.0 + L[23] / 2.0; # e2 ^ (e3 ^ einf)
        Ltemp[23] = (-L[22]) + L[23]; # e2 ^ (e3 ^ e0)
        C[1] = (-(((-(Ptemp[26] * L[21])) + P[28] * Ltemp[19] + (-(P[29] * Ltemp[17]))) * Ptemp[27])) + (-((Ptemp[27] * L[21] + (-(P[28] * Ltemp[20])) + P[29] * Ltemp[18]) * Ptemp[26])) + (-((Ptemp[26] * Ltemp[20] + (-(Ptemp[27] * Ltemp[19])) + P[29] * L[16]) * P[28])) + (-(((-(Ptemp[26] * Ltemp[18])) + Ptemp[27] * Ltemp[17] + (-(P[28] * L[16]))) * P[29])); # e1
        C[2] = (Ptemp[26] * L[24] + (-(P[28] * Ltemp[22])) + P[30] * Ltemp[17]) * Ptemp[27] + ((-(Ptemp[27] * L[24])) + P[28] * Ltemp[23] + (-(P[30] * Ltemp[18]))) * Ptemp[26] + ((-(Ptemp[26] * Ltemp[23])) + Ptemp[27] * Ltemp[22] + (-(P[30] * L[16]))) * P[28] + (-(((-(Ptemp[26] * Ltemp[18])) + Ptemp[27] * Ltemp[17] + (-(P[28] * L[16]))) * P[30])); # e2
        C[3] = (-(((-(Ptemp[26] * L[25])) + P[29] * Ltemp[22] + (-(P[30] * Ltemp[19]))) * Ptemp[27])) + (-((Ptemp[27] * L[25] + (-(P[29] * Ltemp[23])) + P[30] * Ltemp[20]) * Ptemp[26])) + ((-(Ptemp[26] * Ltemp[23])) + Ptemp[27] * Ltemp[22] + (-(P[30] * L[16]))) * P[29] + (Ptemp[26] * Ltemp[20] + (-(Ptemp[27] * Ltemp[19])) + P[29] * L[16]) * P[30]; # e3
        C[4] = (-((P[28] * L[25] + (-(P[29] * L[24])) + P[30] * L[21]) * Ptemp[26])) + (-(((-(Ptemp[26] * L[25])) + P[29] * Ltemp[22] + (-(P[30] * Ltemp[19]))) * P[28])) + (-((Ptemp[26] * L[24] + (-(P[28] * Ltemp[22])) + P[30] * Ltemp[17]) * P[29])) + (-(((-(Ptemp[26] * L[21])) + P[28] * Ltemp[19] + (-(P[29] * Ltemp[17]))) * P[30])); # einf
        C[5] = (-((P[28] * L[25] + (-(P[29] * L[24])) + P[30] * L[21]) * Ptemp[27])) + (Ptemp[27] * L[25] + (-(P[29] * Ltemp[23])) + P[30] * Ltemp[20]) * P[28] + ((-(Ptemp[27] * L[24])) + P[28] * Ltemp[23] + (-(P[30] * Ltemp[18]))) * P[29] + (Ptemp[27] * L[21] + (-(P[28] * Ltemp[20])) + P[29] * Ltemp[18]) * P[30]; # e0
        C[16] = (Ptemp[26] * L[16] + P[28] * Ltemp[17] + P[29] * Ltemp[19] + P[30] * Ltemp[22]) * Ptemp[27] + (Ptemp[27] * L[16] + (-(P[28] * Ltemp[18])) + (-(P[29] * Ltemp[20])) + (-(P[30] * Ltemp[23]))) * Ptemp[26] + ((-(Ptemp[26] * Ltemp[23])) + Ptemp[27] * Ltemp[22] + (-(P[30] * L[16]))) * P[30] + (-((Ptemp[26] * Ltemp[20] + (-(Ptemp[27] * Ltemp[19])) + P[29] * L[16]) * P[29])) + ((-(Ptemp[26] * Ltemp[18])) + Ptemp[27] * Ltemp[17] + (-(P[28] * L[16]))) * P[28]; # e1 ^ (e2 ^ e3)
        C[17] = (Ptemp[26] * Ltemp[18] + Ptemp[27] * Ltemp[17] + (-(P[29] * L[21])) + (-(P[30] * L[24]))) * Ptemp[26] + (Ptemp[26] * L[16] + P[28] * Ltemp[17] + P[29] * Ltemp[19] + P[30] * Ltemp[22]) * P[28] + (-((Ptemp[26] * L[24] + (-(P[28] * Ltemp[22])) + P[30] * Ltemp[17]) * P[30])) + ((-(Ptemp[26] * L[21])) + P[28] * Ltemp[19] + (-(P[29] * Ltemp[17]))) * P[29] + (-(((-(Ptemp[26] * Ltemp[18])) + Ptemp[27] * Ltemp[17] + (-(P[28] * L[16]))) * Ptemp[26])); # e1 ^ (e2 ^ einf)
        C[18] = (Ptemp[26] * Ltemp[18] + Ptemp[27] * Ltemp[17] + (-(P[29] * L[21])) + (-(P[30] * L[24]))) * Ptemp[27] + (-((Ptemp[27] * L[16] + (-(P[28] * Ltemp[18])) + (-(P[29] * Ltemp[20])) + (-(P[30] * Ltemp[23]))) * P[28])) + ((-(Ptemp[27] * L[24])) + P[28] * Ltemp[23] + (-(P[30] * Ltemp[18]))) * P[30] + (-((Ptemp[27] * L[21] + (-(P[28] * Ltemp[20])) + P[29] * Ltemp[18]) * P[29])) + ((-(Ptemp[26] * Ltemp[18])) + Ptemp[27] * Ltemp[17] + (-(P[28] * L[16]))) * Ptemp[27]; # e1 ^ (e2 ^ e0)
        C[19] = (-(((-(Ptemp[26] * Ltemp[20])) + (-(Ptemp[27] * Ltemp[19])) + (-(P[28] * L[21])) + P[30] * L[25]) * Ptemp[26])) + (Ptemp[26] * L[16] + P[28] * Ltemp[17] + P[29] * Ltemp[19] + P[30] * Ltemp[22]) * P[29] + ((-(Ptemp[26] * L[25])) + P[29] * Ltemp[22] + (-(P[30] * Ltemp[19]))) * P[30] + (-(((-(Ptemp[26] * L[21])) + P[28] * Ltemp[19] + (-(P[29] * Ltemp[17]))) * P[28])) + (Ptemp[26] * Ltemp[20] + (-(Ptemp[27] * Ltemp[19])) + P[29] * L[16]) * Ptemp[26]; # e1 ^ (e3 ^ einf)
        C[20] = (-(((-(Ptemp[26] * Ltemp[20])) + (-(Ptemp[27] * Ltemp[19])) + (-(P[28] * L[21])) + P[30] * L[25]) * Ptemp[27])) + (-((Ptemp[27] * L[16] + (-(P[28] * Ltemp[18])) + (-(P[29] * Ltemp[20])) + (-(P[30] * Ltemp[23]))) * P[29])) + (-((Ptemp[27] * L[25] + (-(P[29] * Ltemp[23])) + P[30] * Ltemp[20]) * P[30])) + (Ptemp[27] * L[21] + (-(P[28] * Ltemp[20])) + P[29] * Ltemp[18]) * P[28] + (-((Ptemp[26] * Ltemp[20] + (-(Ptemp[27] * Ltemp[19])) + P[29] * L[16]) * Ptemp[27])); # e1 ^ (e3 ^ e0)
        C[21] = (-(((-(Ptemp[26] * Ltemp[20])) + (-(Ptemp[27] * Ltemp[19])) + (-(P[28] * L[21])) + P[30] * L[25]) * P[28])) + (-((Ptemp[26] * Ltemp[18] + Ptemp[27] * Ltemp[17] + (-(P[29] * L[21])) + (-(P[30] * L[24]))) * P[29])) + (-((P[28] * L[25] + (-(P[29] * L[24])) + P[30] * L[21]) * P[30])) + ((-(Ptemp[26] * L[21])) + P[28] * Ltemp[19] + (-(P[29] * Ltemp[17]))) * Ptemp[27] + (-((Ptemp[27] * L[21] + (-(P[28] * Ltemp[20])) + P[29] * Ltemp[18]) * Ptemp[26])); # e1 ^ (einf ^ e0)
        C[22] = (Ptemp[26] * Ltemp[23] + Ptemp[27] * Ltemp[22] + P[28] * L[24] + P[29] * L[25]) * Ptemp[26] + (Ptemp[26] * L[16] + P[28] * Ltemp[17] + P[29] * Ltemp[19] + P[30] * Ltemp[22]) * P[30] + (-(((-(Ptemp[26] * L[25])) + P[29] * Ltemp[22] + (-(P[30] * Ltemp[19]))) * P[29])) + (Ptemp[26] * L[24] + (-(P[28] * Ltemp[22])) + P[30] * Ltemp[17]) * P[28] + (-(((-(Ptemp[26] * Ltemp[23])) + Ptemp[27] * Ltemp[22] + (-(P[30] * L[16]))) * Ptemp[26])); # e2 ^ (e3 ^ einf)
        C[23] = (Ptemp[26] * Ltemp[23] + Ptemp[27] * Ltemp[22] + P[28] * L[24] + P[29] * L[25]) * Ptemp[27] + (-((Ptemp[27] * L[16] + (-(P[28] * Ltemp[18])) + (-(P[29] * Ltemp[20])) + (-(P[30] * Ltemp[23]))) * P[30])) + (Ptemp[27] * L[25] + (-(P[29] * Ltemp[23])) + P[30] * Ltemp[20]) * P[29] + (-(((-(Ptemp[27] * L[24])) + P[28] * Ltemp[23] + (-(P[30] * Ltemp[18]))) * P[28])) + ((-(Ptemp[26] * Ltemp[23])) + Ptemp[27] * Ltemp[22] + (-(P[30] * L[16]))) * Ptemp[27]; # e2 ^ (e3 ^ e0)
        C[24] = (Ptemp[26] * Ltemp[23] + Ptemp[27] * Ltemp[22] + P[28] * L[24] + P[29] * L[25]) * P[28] + (-((Ptemp[26] * Ltemp[18] + Ptemp[27] * Ltemp[17] + (-(P[29] * L[21])) + (-(P[30] * L[24]))) * P[30])) + (P[28] * L[25] + (-(P[29] * L[24])) + P[30] * L[21]) * P[29] + (-((Ptemp[26] * L[24] + (-(P[28] * Ltemp[22])) + P[30] * Ltemp[17]) * Ptemp[27])) + ((-(Ptemp[27] * L[24])) + P[28] * Ltemp[23] + (-(P[30] * Ltemp[18]))) * Ptemp[26]; # e2 ^ (einf ^ e0)
        C[25] = (Ptemp[26] * Ltemp[23] + Ptemp[27] * Ltemp[22] + P[28] * L[24] + P[29] * L[25]) * P[29] + ((-(Ptemp[26] * Ltemp[20])) + (-(Ptemp[27] * Ltemp[19])) + (-(P[28] * L[21])) + P[30] * L[25]) * P[30] + (-((P[28] * L[25] + (-(P[29] * L[24])) + P[30] * L[21]) * P[28])) + ((-(Ptemp[26] * L[25])) + P[29] * Ltemp[22] + (-(P[30] * Ltemp[19]))) * Ptemp[27] + (-((Ptemp[27] * L[25] + (-(P[29] * Ltemp[23])) + P[30] * Ltemp[20]) * Ptemp[26])); # e3 ^ (einf ^ e0)
        C[31] = (Ptemp[26] * Ltemp[23] + Ptemp[27] * Ltemp[22] + P[28] * L[24] + P[29] * L[25]) * P[30] + (-(((-(Ptemp[26] * Ltemp[20])) + (-(Ptemp[27] * Ltemp[19])) + (-(P[28] * L[21])) + P[30] * L[25]) * P[29])) + (Ptemp[26] * Ltemp[18] + Ptemp[27] * Ltemp[17] + (-(P[29] * L[21])) + (-(P[30] * L[24]))) * P[28] + (-((Ptemp[26] * L[16] + P[28] * Ltemp[17] + P[29] * Ltemp[19] + P[30] * Ltemp[22]) * Ptemp[27])) + (Ptemp[27] * L[16] + (-(P[28] * Ltemp[18])) + (-(P[29] * Ltemp[20])) + (-(P[30] * Ltemp[23]))) * Ptemp[26]; # e1 ^ (e2 ^ (e3 ^ (einf ^ e0)))
        return gaalop_map@C
    gmt_func = layout.gmt_func
    @numba.njit
    def native_reflect(P_val,L_val):
        return gmt_func(gmt_func(P_val,L_val),P_val)

    print('\n\nSTARTINGTEST\n')
    P1 = layout.randomMV()(4)
    L1 = layout.randomMV()(3)
    import time
    start_time = time.time()
    for i in range(NTESTS):
        C = gaalop(P1.value,L1.value)
    print('gaalop time: ', time.time() - start_time)
    start_time = time.time()
    for i in range(NTESTS):
        C =native_reflect(P1.value,L1.value)
    print('native time: ', time.time() - start_time)
