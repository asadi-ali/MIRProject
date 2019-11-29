

def gamma_compress(integers):
    result = ''
    for integer in integers:
        if not integer:
            return ''

        res = bin(integer)[3:]
        result += '1' * len(res) + '0' + res

    return result


def gamma_uncompress(string):
    result = []
    while string.find('0') != -1:
        length = string.find('0')
        result.append(int('0b1' + string[length+1:2*length+1], 2))
        string = string[2*length+1:]
    return result
