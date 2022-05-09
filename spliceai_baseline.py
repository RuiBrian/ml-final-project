from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np




# set whether file is donor sequences (True) or acceptor (False)
def calc_acc(file, is_donor):
    seqs = []
    with open(file, 'r') as f:
        for line in f:
            seqs.append(line.strip())

    corrects = []
    i = 0
    mid = len(seqs[0]) / 2

    for s in seqs:
        i += 1
        print(i)
        input_sequence = s
        # Replace this with your custom sequence
        context = 10000
        paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
        models = [load_model(resource_filename('spliceai', x), compile=False) for x in paths]
        x = one_hot_encode('N' * (context // 2) + input_sequence + 'N' * (context // 2))[None, :]
        y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

        acceptor_prob = y[0, :, 1]
        donor_prob = y[0, :, 2]
        neither_prob = y[0, :, 0]


        amax = np.where(acceptor_prob == np.max(acceptor_prob))[0][0]

        dmax = np.where(donor_prob == np.max(donor_prob))[0][0]

        # for donors dmax == 39, for acceptors amax == 42
        # for donors dmax == 199, for acceptors amax == 202

        if is_donor:
            if dmax == (mid - 2):
                corrects.append(1)
                print(1)
            else:
                corrects.append(0)
                print(0)

        else:
            if amax == (mid + 1):
                corrects.append(1)
                print(1)
            else:
                corrects.append(0)
                print(0)

    print(sum(corrects))
