def reader(inputs, hps):
    inputs1 = []
    with open("data1/file1.txt") as f:
        for line in f:
            if line[:-1]:
                inputs1.append(float(line[:-1]))

    inputs2 = []
    with open("data1/file2.txt") as f:
        for line in f:
            if line[:-1]:
                inputs2.append(float(line[:-1]))

    inputs3 = []
    with open("data2/file1.txt") as f:
        for line in f:
            if line[:-1]:
                inputs3.append(float(line[:-1]))

    inputs4 = []
    with open("data2/file2.txt") as f:
        for line in f:
            if line[:-1]:
                inputs4.append(float(line[:-1]))

    ret = []
    for x in inputs1 + inputs2 + inputs3 + inputs4:
        x = x * hps['mult'] + hps['add']
        ret.append(x)
    return ret
    
