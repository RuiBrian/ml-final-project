import random
import numpy as np
acceptors_dict = {}
donors_dict = {}
flankingbps = 200
exons = {}


with open('spliceai_introns.txt','r') as f:
    for line in f:
        if not line.startswith('#'):
            l = line.strip()
            s = l.split('\t')
            acceptors= s[5].split(',')
            flankedA = []
            for i in range(1, len(acceptors)-1):
                a = int(acceptors[i])
                start = a -flankingbps -2
                end = a + flankingbps
                flankedA.append([start, end])


            donors = s[6].split(',')
            flankedD = []
            for i in range(len(donors)-2):
                a = int(donors[i])
                start = a -  flankingbps
                end = a + 2  + flankingbps
                flankedD.append([start, end])

            info = s[0]+ ';chr' + s[1] + ';' + s[2]
            if flankedA:
                acceptors_dict[info] = flankedA
                donors_dict[info] = flankedD
            
                cur_exons = []
                starts = s[5].split(',')
                ends = s[6].split(',') 
                
                for i in range(len(starts) -1): 
                    start = int(starts[i])
                    end = int(ends[i])
                    
                    if end > (start + 2* flankingbps):
                        randEx = random.randint(start+ flankingbps, end - flankingbps) 
                        cur_exons.append([randEx- flankingbps, randEx + 2 + flankingbps])
                exons[info] = cur_exons

all = []
with open('acceptors_test.bed', 'w') as out:
    for k in acceptors_dict.keys():
        x = k.split(';')
        temp = acceptors_dict[k]

        for vals in temp:
            all.append(vals[0])
            all.append(vals[1])
            out.write(x[1] + '\t' + str(vals[0]) + '\t' + str(vals[1]) + '\t' + x[0] + '\t' + '.' + '\t' + x[2] + '\n')


with open('donors_test.bed', 'w') as out:
    for k in donors_dict.keys():
        x = k.split(';')
        temp = donors_dict[k]
        for  vals in temp:
            all.append(vals[0])
            all.append(vals[1])
            out.write(x[1] + '\t' + str(vals[0]) + '\t' + str(vals[1]) + '\t' + x[0] + '\t' + '.' + '\t' + x[2] + '\n')


with open('exons_test.bed', 'w') as out:
    for k in exons.keys():
        x = k.split(';')
        temp = exons[k]
        for  vals in temp:
            all.append(vals[0])
            all.append(vals[1])
            out.write(x[1] + '\t' + str(vals[0]) + '\t' + str(vals[1]) + '\t' + x[0] + '\t' + '.' + '\t' + x[2] + '\n')


min = np.min(all)
max = np.max(all)
randomlist = random.sample(range(10000, 46709983), 200000)

with open('neither_test.bed', 'w') as out:
    for ran in randomlist:
        
        n = random.randint(1, 21)
        chrom = 'chr' + str(n) 
        end = ran + 2 + flankingbps
        start = ran - flankingbps
        out.write(chrom + '\t' + str(start) + '\t' + str(end) + '\n')


