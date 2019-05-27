import methods
from fasta_reader import FASTA
import numpy as np
newstr = []
sign = 1
title = ""
def generator(input_file_name, feature_file_name, feature_list):
    order, sequences = FASTA(input_file_name)
    print ("-> Window feature set generating ...");
    for s in order:
        if(s[0]=='p'):
            sign = 1
        else:
            sign = 0
        p = sequences[s]
        each_feature_vector = ""

        #1#4
        if(feature_list[0]):
            a, c, g, t = methods.frequence_count(p, 'A', 'C', 'G', 'T')
            each_feature_vector = each_feature_vector + "%d," % (g + c)

        #2#24
        if(feature_list[1]):
            a, c, g, t = methods.frequence_count(p[45:55], 'A', 'C', 'G', 'T')
            each_feature_vector = each_feature_vector + "%d," % (g + c)

        #3#105
        if(feature_list[2]):
            if p[59] == 'G':
                value = 1
            elif s == 'C':
                value = -1
            else:
                value = 0
            each_feature_vector = each_feature_vector + "%d," % value

        #4#107
        if(feature_list[3]):
            if p[61] == 'G':
                value = 1
            elif s == 'C':
                value = -1
            else:
                value = 0
            each_feature_vector = each_feature_vector + "%d," % value


        #5#171
        if(feature_list[4]):
            sq = p[45:55]
            cnt = 0
            for i in range(len(sq)-1):
                if (sq[i]=='T') and (sq[i+1]=='A'):
                    cnt += 1

            each_feature_vector = each_feature_vector + "%d," % cnt


        #6#237
        if(feature_list[5]):
            sq = p
            cnt = 0
            for i in range(len(sq) - 2):
                if (sq[i] == 'T') and (sq[i + 1] == 'T') and (sq[i + 2] == 'G'):
                    cnt += 1

            each_feature_vector = each_feature_vector + "%d," % cnt

        #7#267

        if(feature_list[6]):
            sq = p[20:35]
            cnt = 0
            for i in range(len(sq) - 2):
                if (sq[i] == 'C') and (sq[i + 1] == 'T') and (sq[i + 2] == 'A'):
                    cnt += 1

            each_feature_vector = each_feature_vector + "%d," % cnt


        #8#3152
        if(feature_list[7]):
            sq = p[20:35]
            cnt = 0
            for i in range(len(sq) - 4):
                if (sq[i] == 'T') and (sq[i + 1] == 'T') and (sq[i + 2] == 'G') and (sq[i + 3] == 'A') and (sq[i + 4] == 'C'):
                    cnt += 1

            each_feature_vector = each_feature_vector + "%d," % cnt

        #9#16628
        if(feature_list[8]):
            cnt = 0
            g = 10
            sq = p
            for i in range(len(sq)-(g+1)):
                if (sq[i]=='C') and (sq[i+(g+1)]=='C'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #10#17055
        if(feature_list[9]):
            cnt = 0
            g = 1
            sq = p[45:55]
            for i in range(len(sq) - (g+1)):
                if (sq[i] == 'A') and (sq[i + (g+1)] == 'A'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt

        #11#17375
        if(feature_list[10]):
            cnt = 0
            g = 4
            sq = p
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'C') and (sq[i+1]=='A') and (sq[i + (g+2)] == 'A'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt

        #12#17458
        if(feature_list[11]):
            cnt = 0
            g = 5
            sq = p
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'G') and (sq[i + 1] == 'A') and (sq[i + (g+2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #13#17602
        if(feature_list[12]):
            cnt = 0
            g = 7
            sq = p
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'T') and (sq[i + 1] == 'A') and (sq[i + (g+2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #14#17886
        if(feature_list[13]):
            cnt = 0
            g = 12
            sq = p
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'A') and (sq[i + 1] == 'T') and (sq[i + (g+2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #15#18510
        if(feature_list[14]):
            cnt = 0
            g = 21
            sq = p
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'T') and (sq[i + 1] == 'T') and (sq[i + (g+2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #16#18890
        if(feature_list[15]):
            cnt = 0
            g = 3
            sq = p[20:35]
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'T') and (sq[i + 1] == 'G') and (sq[i + (g+2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt

        #17#19188
        if(feature_list[16]):
            cnt = 0
            g = 8
            sq = p[20:35]
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'G') and (sq[i + 1] == 'C') and (sq[i + (g+2)] == 'C'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt

        #18#19586
        if(feature_list[17]):
            cnt = 0
            g = 3
            sq = p[45:55]
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'T') and (sq[i + 1] == 'A') and (sq[i + (g+2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt

        #19#19993
        if(feature_list[18]):
            cnt = 0
            g = 3
            sq = p
            for i in range(len(sq) - (g+2)):
                if (sq[i] == 'A') and (sq[i+(g+1)] == 'C') and (sq[i + (g+2)] == 'G'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #20#20492
        if(feature_list[19]):
            cnt = 0
            g = 10
            sq = p
            for i in range(len(sq) - (g + 2)):
                if (sq[i] == 'T') and (sq[i + (g + 1)] == 'T') and (sq[i + (g + 2)] == 'C'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #21#20826
        if(feature_list[20]):
            cnt = 0
            g = 16
            sq = p
            for i in range(len(sq) - (g + 2)):
                if (sq[i] == 'A') and (sq[i + (g + 1)] == 'G') and (sq[i + (g + 2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #22#22162
        if(feature_list[21]):
            cnt = 0
            g = 2
            sq = p[45:55]
            for i in range(len(sq) - (g + 2)):
                if (sq[i] == 'A') and (sq[i + (g + 1)] == 'A') and (sq[i + (g + 2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        #23#22166
        if(feature_list[22]):
            cnt = 0
            g = 2
            sq = p[45:55]
            for i in range(len(sq) - (g + 2)):
                if (sq[i] == 'A') and (sq[i + (g + 1)] == 'C') and (sq[i + (g + 2)] == 'T'):
                    cnt += 1
            each_feature_vector = each_feature_vector + "%d," % cnt


        if(feature_list[23]):
            sq = p
            threshold = 3
            #24#22543
            each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(sq[45:55], "TATAAT", threshold))

        #25#22563
        if(feature_list[24]):
            each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(sq[45:55], "TTATAA", threshold))

        #26#22567
        if(feature_list[25]):
            each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(sq[20:35], "TTGACA", threshold))

        #27#22594
        if(feature_list[26]):
            each_feature_vector = each_feature_vector + "%f," % methods.dinucleotide_value(p[20:35])


        each_feature_vector = each_feature_vector+"%d"%sign
        # For combining all Features
        newstr.append(each_feature_vector)

    print ('-> '+feature_file_name +" creating  ...");
    file_object = open(feature_file_name,"w+")
    for p in newstr:
        file_object.writelines(p+"\n")

    file_object.close()
    print ("-> Complete Features Set  ...");