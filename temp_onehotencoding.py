import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dir_data = "/Users/carolynayamamoto/Documents/GitHub/cs475/finalproject/ml-final-project/datasets/"

for filename in os.listdir(dir_data):
    file = os.path.join(dir_data, filename)
    with open(file) as fasta_file:
        fasta_contents = fasta_file.readlines()
        # for line in fasta_contents:
        #     print(line, '\n')

print(fasta_contents[5])

line = []
line[:] = fasta_contents[5][0:82]
print(line)
print(len(line))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(line)
print(integer_encoded)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)
