#use command to get sequence from bed file in previous step as input
#bedtools getfasta -s -fi hg38_p8.fa -bed gencode.v38_latest_locs3.gff > test_400.fa

dict = {}
flank = 200
with open('test_400.fa','r') as f:
    prevline = ''
    for line in f:
        l = line.strip()
        if l.startswith('>'):
            prevline = l

        else:
            #get donors and acceptors at beginning and end of intron sequence with flanking bps
            first = l[:(2*flank+2)].upper()
            last = l[-(2*flank +2):].upper()
           
            dict[prevline] = [first,last]

#write ouput file a id , donor seq, acceptor setq
with open('seqs_400.txt','w') as out:
    for k in dict.keys():
        loc = k #.split(':')
        #finLoc = loc[0] + '_' + loc[-1]

        values = dict[k]
        out.write(loc + '\t' + values[0] + '\t' + values[1] + '\n')
