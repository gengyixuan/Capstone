def writer(inputs, hps):
    ret = {}
    ret['result_list'] = []
    ret['sum_of_stuff'] = 0
    for line in inputs['append1']:
        x, ap = line
        ret['result_list'].append((x-hps['subtract'], ap, hps['classifier']))
        ret['sum_of_stuff'] += abs(x-hps['subtract'])
    return ret