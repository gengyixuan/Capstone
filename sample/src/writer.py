def writer(inputs, hps):
    ret = []
    for line in inputs['append1']:
        x, ap = line
        ret.append((x-hps['subtract'], ap, hps['classifier']))
    for line in inputs['append2']:
        x, ap1, ap2 = line
        ret.append((x-hps['subtract'], ap1, ap2, hps['classifier']))
    return ret