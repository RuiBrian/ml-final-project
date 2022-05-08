from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

input_sequence = 'CTTCGTGGGGGGCCTGTCTGGCTGTGGAGAGTACACTCGGGTAAGGGGGGGCCCCAGTTCCTGGGGCGGGGCTGGAGCTGGC'
#'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
#"AAAAAAAAAAAAATCTGGGTTGCACGTGTAAGGCTGGCCCTAGCTAAGGAGCGTGAAGTACAGAAATTTTCAGGTCATCTCA"
#"AAAAAAAAAAAAACTGGAGTACTAGCCTGGCTTACTTTGCAGGAGTGCCTTCTCTAGAACTGCTAGCATGTCAGATCAGCTC"
#'TTTTTCGTAAACAGCAAAAA' #first GT donor, last AG acceptor

context = 10000
nummodels = 6
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, nummodels))
# paths = ('spliceaimodels/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x),compile=False) for x in paths]
# models = [load_model(x,compile=False) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(nummodels-1)], axis=0)

neither_prob = y[0, :, 0]
acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]

print(neither_prob)
nmin = np.where(neither_prob==np.min(neither_prob))[0][0]
print(f"neither min index: {nmin}")
print(f"acceptor: {acceptor_prob}")
amax = np.where(acceptor_prob==np.max(acceptor_prob))[0][0]
print(f"acceptor max index: {amax}")
print(f"acceptor len{len((acceptor_prob))} and input len {len(input_sequence)}")
print(f"donor: {donor_prob}")
dmax = np.where(donor_prob==np.max(donor_prob))[0][0]
print(f"donor max index: {dmax}")
print(f"donor: {input_sequence[dmax+1:dmax+3]}")
print(f"acceptor: {input_sequence[amax-2:amax]}")

