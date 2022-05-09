from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np


# flank is either 40 or 200
# set whether donor sequences or not
def calc_acc(sequences, is_donor, flank):
    corrects = []
    i = 0

    for s in sequences:
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

        nmin = np.where(neither_prob == np.min(neither_prob))[0][0]

        amax = np.where(acceptor_prob == np.max(acceptor_prob))[0][0]

        dmax = np.where(donor_prob == np.max(donor_prob))[0][0]

        # for donors dmax == 39, for acceptors amax == 42
        # for donors dmax == 199, for acceptors amax == 202

        if is_donor:
            if dmax == (flank - 1):
                corrects.append(1)
                print(1)
            else:
                corrects.append(0)
                print(0)

        else:
            if amax == (flank + 2):
                corrects.append(1)
                print(1)
            else:
                corrects.append(0)
                print(0)

    print(sum(corrects))
