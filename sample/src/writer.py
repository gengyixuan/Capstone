def writer(inputs, hps):
    ret = {}
    ret['result_list'] = []
    ret['sum_of_stuff'] = 0
    for line in inputs['append1']:
        x, ap = line
        ret['result_list'].append((x-hps['subtract'], ap))
        ret['sum_of_stuff'] += abs(x-hps['subtract'])
    for line in inputs['append2']:
        x, ap1, ap2 = line
        ret['result_list'].append((x-hps['subtract'], ap1, ap2))
        ret['sum_of_stuff'] += abs(x-hps['subtract'])
    return ret