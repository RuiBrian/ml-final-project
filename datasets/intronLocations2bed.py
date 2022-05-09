dict= {}
#set flanking sequence desired from either side
flank = 200
with open('gencode.v38_latest.gff','r') as f:
    with open('gencode.v38_latest_locs3.gff','w') as out:
        exons = []
        #store all exons for a given parent
        for line in f:
            if not line.startswith('#'):
                s = line.split('\t')
                typ = s[2]
                if typ == 'exon':
                    p = s[-1].strip()
                    coms = p.split(';')

                    parent = ''
                    for c in coms:
                        if c.startswith('Parent'):
                            fullid = c
                            full = fullid.split('=')
                            parent = full[1]
                            break

                    if parent in dict.keys():
                        val = dict[parent]
                        val.append(line)
                        dict[parent] = val
                    else:
                        dict[parent] = [line]

        #loop through all exons of a parent and calculate introns i
        #create an output bed file of intron  locations woth flanking sequence on either side
        for transcript in dict.keys():
            values = dict[transcript]
            if len(values) > 1:
                count = 0

                for i in range(1, len(values)):

                    exonline = values[i]
                    prevexonLine = values[i-1]
                    sprevexon = prevexonLine.split('\t')
                    sexon = exonline.split('\t')
                    intronStartBed = int(sprevexon[4]) + 1 - flank 
                    intronEndBed = int(sexon[3]) - 1 + flank

                    if (intronEndBed - intronStartBed) < 4:
                        continue
                    count += 1
                    comments = sexon[-1].split(';')
                    chr = s[0]

                    for c in comments:
                        c = c.strip()
                        if c.startswith('Parent'):
                            full = c.split('=')
                            parent = full[1]
                    intronID = parent + '_' + str(count)
                    temp = sexon
                    temp[3] = str(intronStartBed)
                    temp[4] = str(intronEndBed)

                    temp[-1] = 'ID='+ intronID
                    if transcript == 'ALL_10139197':
                        print(temp)
                    #out.write( chr + '\t' + intronStartBed + '\t' + intronEndBed +'\t' +intronID + '\n')
                    for t in temp:
                        out.write(t + '\t')
                    out.write('\n')


