def writer(inputs, hps):
    ret = []
    for line in inputs['append1']:
        x, ap = line
        ret.append((x-hps['subtract'], ap, hps['classifier']))
    return ret