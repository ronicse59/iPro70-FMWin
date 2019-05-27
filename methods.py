import numpy as np
def frequence_count(p,v1,v2,v3,v4):
    a = p.count(v1)
    c = p.count(v2)
    g = p.count(v3)
    t = p.count(v4)
    return a,c,g,t

# Two mar count
def two_mar_frequency_count(p):
    vector = ""
    sum = 0.0
    DNA = 'ACGT'
    for a in DNA:
        for b in DNA:
            combination = a + b
            vector = vector + "%d," % p.count(combination)
    return vector

# Three mar count
def three_mar_frequency_count(p):
    vector = ""
    DNA = 'ACGT'
    for a in DNA:
        for b in DNA:
            for c in DNA:
                combination = a + b + c
                vector = vector + "%d," % p.count(combination)
    return vector

def four_mar_frequency_count(p):
    vector = ""
    DNA = 'ACGT'
    for a in DNA:
        for b in DNA:
            for c in DNA:
                for d in DNA:
                    combination = a + b + c + d
                    vector = vector + "%d," % p.count(combination)
    return vector

def five_mar_frequency_count(p):
    vector = ""
    DNA = 'ACGT'
    for a in DNA:
        for b in DNA:
            for c in DNA:
                for d in DNA:
                    for e in DNA:
                        combination = a + b + c + d + e
                        vector = vector + "%d," % p.count(combination)
    return vector

def six_mar_frequency_count(p):
    vector = ""
    DNA = 'ACGT'
    for a in DNA:
        for b in DNA:
            for c in DNA:
                for d in DNA:
                    for e in DNA:
                        for f in DNA:
                            combination = a + b + c + d + e + f
                            vector = vector + "%d," % p.count(combination)
    return vector

# Scaling character into int
def scl(p):
    if p == 'A':
        return 0
    elif p == 'C':
        return 1
    elif p == 'G':
        return 2
    elif p == 'T':
        return 3
# Function for A_A,A_C,A_G,A_T,C_C,C_A,C_G,C_T .... T_T
def two_mar_k_gap(str, k_gap):
    vector = ""
    cnt = np.zeros((4, 4), dtype=int)
    for i in range(len(str) - (k_gap+1)):
        x = scl(str[i])
        y = scl(str[i + (k_gap+1)])
        cnt[x][y] += 1

    for x in range(4):
        for y in range(4):
            vector = vector + ("%d," % cnt[x, y])
    return vector

# For FeatureSet 3: AA_A, AA_C, AA_G ....TT_T
def three_mar_right_k_gap(str,k_gap):
    vector = ""
    cnt = np.zeros((4, 4, 4), dtype=int)
    for i in range(len(str) - (k_gap + 2)):
        x = scl(str[i])
        y = scl(str[i + 1])
        z = scl(str[i + (k_gap + 2)])
        cnt[x][y][z] += 1

    for x in range(4):
        for y in range(4):
            for z in range(4):
                vector = vector + ("%d," % cnt[x, y, z])
    return vector

# For FeatureSet 3: A_AA, A_AC, A_AG ....T_TT
def three_mar_left_k_gap(str,k_gap):
    vector = ""
    cnt = np.zeros((4, 4, 4), dtype=int)
    for i in range(len(str) - (k_gap + 2)):
        x = scl(str[i])
        y = scl(str[i + (k_gap + 1)])
        z = scl(str[i + (k_gap + 2)])
        cnt[x][y][z] += 1

    for x in range(4):
        for y in range(4):
            for z in range(4):
                vector = vector + ("%d," % cnt[x, y, z])
    return vector

# Substring finding based on missing value threashold
def string_matching(str1, str2, threshold):
    distance = 0
    for i in range(len(str1)-len(str2)):
        cnt = 0
        for j in range(0, len(str2)):
            if(str1[i+j] == str2[j]):
                cnt += 1
        if (cnt>distance) and (cnt > threshold):
            distance = cnt

    return distance

# Character distance count
def distance_count(string, char):
    index = []
    cnt = 0
    for i in range(len(string)):
        if string[i] == char:
            index.append(i)

    for i in range(1, len(index)):
        cnt += (index[i] - index[i - 1])
    return cnt

def numerical_position(string):
    string = string.replace('A', '1,')
    string = string.replace('C', '2,')
    string = string.replace('G', '3,')
    string = string.replace('T', '4,')
    return string

# Dinucleotide Parameters Based on DNasel Digestion Data.
# "Trinucleotide Models for DNA Bending Propensity:
# Comparison of Models Based on DNasei Digestion
# and Nucleosome Packaging Data"
DNaseI_drived = np.zeros((4, 4), dtype=int)
                    #AA     AC      AG      AT
DNaseI_drived = [[-0.227, -0.051, -0.010, -0.026],
                    #CA     CC      CG      CT
                   [0.130, -0.038, -0.129, -0.010],
                    #GA     GC      GG      GT
                   [0.059, 0.084, -0.038, -0.051],
                    #TA     TC      TG      TT
                   [0.213, 0.059, 0.130, -0.227]]
Wedge_roll_bolshoy = np.zeros((4, 4), dtype=int)
                        #AA   AC   AG    AT
Wedge_roll_bolshoy = [[-6.5, -0.9, 8.4, 2.6],
                        #GA   GC   GG    GT
                       [1.6, 1.2, 6.7, 8.4],
                        #GA   GC   GG    GT
                       [-2.7, -5.0, 1.2, -0.9],
                        #TA   TC   TG    TT
                       [0.9, -2.7, 1.6, -6.5]]

roll_de_santis = np.zeros((4, 4), dtype=int)
                    #AA   AC    AG    AT
roll_de_santis = [[-5.4, -2.4, 1.0, -7.3],
                    #CA   CC   CG    CT
                   [6.7, 1.3, 4.6, 1.0],
                    #GA   GC   GG    GT
                   [2.0, -3.7, 1.3, -2.4],
                    #TA   TC   TG    TT
                   [8.0, 2.0, 6.7, -5.4]]

# roll_gartenberg_crothers = np.zeros((4, 4), dtype=int)
                              #AA    AC     AG     AT
roll_gartenberg_crothers = [[1.135, 1.040, 1.050, 1.120],
                              #CA    CC     CG     CT
                           [1.045, 0.995, 1.020, 1.050],
                              #GA    GC     GG     GT
                           [1.055, 0.980, 0.995, 1.040],
                              #TA    TC     TG     TT
                           [1.070, 1.055, 1.045, 1.135]]
def dinucleotide_value(p):
    val1 = val2 = val3 = val4 = 0
    for i in range(1, len(p)):
        a = scl(p[i - 1])
        b = scl(p[i])
        val1 = val1 + DNaseI_drived[a][b]
        val2 = val2 + Wedge_roll_bolshoy[a][b]
        val3 = val3 + roll_de_santis[a][b]
        val4 = val4 + roll_gartenberg_crothers[a][b]
    return val1

# Code likhte hobe      #AAA
DNaseI_based_model = [[[-0.274, -0.205, -0.081, -0.280],[-0.006,-0.032,-0.033,-0.183],[0.027,0.017,-0.057,-0.183],[0.182,-0.110,0.134,-0.280]],
                      [[0.015,0.040,0.175,0.134],[-0.246,-0.012,-0.136,-0.057],[-0.003,-0.077,-0.136,-0.033],[0.090,0.031,0.175,-0.081]],
                      [[-0.037,-0.013,0.031,-0.110],[0.076,0.107,-0.077,0.017],[0.013,0.107,-0.012,-0.032],[0.025,-0.013,0.040,-0.205]],
                      [[0.068,0.025,0.090,0.182],[0.194,0.013,-0.003,0.027],[0.194,0.076,-0.246,-0.006],[0.068,-0.037,0.015,-0.274]]]

def trinucleotide_value(p):
    val1 = 0
    for i in range(2, len(p)):
        a = scl(p[i - 2])
        b = scl(p[i - 1])
        c = scl(p[i])
        val1 = val1 + DNaseI_based_model[a][b][c]
    return val1

# For cross interchange in between two clusters
def cluster_interchange(cluster0, class0, cluster1, class1):
    vector_cluster0 = np.concatenate([cluster0[class0==1], cluster1[class1==0]])
    vector_class0 = np.concatenate([class0[class0 == 1], class1[class1 == 0]])

    vector_cluster1 = np.concatenate([cluster1[class1==1], cluster0[class0==0]])
    vector_class1 = np.concatenate([class1[class1==1], class0[class0==0]])

    return vector_cluster0, vector_class0, vector_cluster1, vector_class1