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
        result.append(int('0b1' + string[length + 1:2 * length + 1], 2))
        string = string[2 * length + 1:]
    return result


def variable_compress(integers):
    result = ''
    for integer in integers:
        res = []
        while integer >= 128:
            res.append(bin(integer % 128)[2:].zfill(7))
            integer //= 128
        res.append(bin(integer)[2:].zfill(7))
        res.reverse()
        for x in res[:-1]:
            result += '0' + x
        result += '1' + res[-1]
    return result


def variable_uncompress(string):
    result = []
    pre = 0
    for i in range(len(string) // 8):
        integer = int(string[i * 8:i * 8 + 8], 2)
        if integer >= 128:
            result.append(pre * 128 + integer - 128)
            pre = 0
        else:
            pre = pre * 128 + integer

    return result
