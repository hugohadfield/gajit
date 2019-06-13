
import re
import hashlib

def process_to_symbols(text, inputs, intermediates, outputs):

    # Get all the symbols
    all_symbols = set(re.findall("[A-Za-z1-9]+\[\d+\]", text))
    input_symbols = set()
    output_symbols = set()
    remapped_symbols = set()

    # All the input symbols can be defined right at the start
    ind = text.find('):') + 2
    for inp in inputs:
        for j in range(32):
            symbol = inp + '\['+str(j)+'\]'
            symbol_text = inp + '[' + str(j) + ']'
            replacement = inp + '_'+str(j)
            text = re.sub(symbol, replacement, text)
            remapped_symbols.add(replacement)
            input_symbols.add(symbol_text)
            text = text[:ind]+'\n    ' +replacement + ' = ' +symbol_text +text[ind:]

    # We can leave all of the output symbols alone
    for inp in outputs:
        for j in range(32):
            symbol = inp + '['+str(j)+']'
            output_symbols.add(symbol)

    # Get all the symbols other than the input, output
    others = all_symbols.difference(input_symbols)
    others = others.difference(output_symbols)

    # For each symbol get the number
    for symb in others:
        splt = symb.split('[')
        tx = splt[0]
        numb = splt[1][:-1]
        replacement = tx+'_'+numb
        remapped_symbols.add(replacement)
        text = text.replace(symb,replacement)

    return common_subexpression_elimination(text)


def common_subexpression_elimination(text):

    # Symbol * Symbol
    # [A-Za-z]\w+\s\*\s[A-Za-z]\w+(?!\.)
    reg_finder = "[A-Za-z]\w+\s\*\s[A-Za-z]\w+(?!\.)"
    text2 = text + ' '

    # Find all matching symbols
    matches = list(set(re.findall(reg_finder, text2)))

    # Find all associated replacements
    replist = ["__".join(("".join(m.split())).split('*')) for m in matches]

    for m, rep in zip(matches, replist):
        # Generate a replacement
        # Find first usage
        x = text2.find(m)
        # Step back to the start of the line
        for i in range(len(text2)):
            ind = x - i
            if text2[ind] == '\n':
                ind = ind + 1
                break
        text2 = text2[:ind] + '    ' + rep + ' = ' + m + '\n' + text2[ind:].replace(m,rep)
    return text2


if __name__ == '__main__':
    with open('example_rotor_lines_gajit.py','r') as fobj:
        text = fobj.read()
        inputs = ['L','M']
        outputs = ['V']
        intermediates = []
        text = process_to_symbols(text, inputs, intermediates, outputs)
        #print(text)
