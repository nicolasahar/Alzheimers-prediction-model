import nltk
import os
import csv
import numpy as np
import math

#export CLASSPATH=/h/u2/c4m/stanford-parser-full-2015-12-09/

from nltk.stem.porter import *
from nltk.tag.stanford import StanfordPOSTagger
from nltk.internals import find_jars_within_path
from nltk.parse.stanford import StanfordParser
from sklearn import svm

def load(file):
    """ (file open for reading) -> list of str

    Return the list of filenames from file, which contains
    one filename per line.
    """

    file_list = []
    for address in file:
        file_list.append(address.strip('\n'))
    return file_list

def preprocess(flist, folder_path):
    """ (file open for reading, str) -> Nonetype

    flist contains one filename per line and folder_path represents a 
    directory. Do preprocessing on each file from flist in folder_path.
    """

    error_log = []
    for i in range(len(flist)):

        path = flist[i]

        stemmer = PorterStemmer()
        parser = StanfordParser(
            model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',
             verbose=True)
        stanford_dir = parser._classpath[0].rpartition('/')[0]
        parser._classpath = tuple(find_jars_within_path(stanford_dir))

        with open(path, 'r') as rf:
            try:
                sent = [line.strip('\n ') for line in rf]
            except UnicodeDecodeError as e:
                error_log.append('Unicode Decode Error:\t' + path + '\n')
                pass
            else:
                if not sent:
                    error_log.append('Empty File Error:\t' + path + '\n')
                    pass
                else:
                    # Stemming with Porter Stemmer
                    pars_stem = stemmer.stem(' '.join(sent))
                    stemmed = '\n'.join(sent)

                    wf = open(folder_path 
                        + path.split('.')[0].split('/')[-1] + '.stem', 'w')
                    wf.write(stemmed)
                    wf.close()

                    # POS Tagging after tokenizing and stemming
                    pos = nltk.pos_tag(pars_stem.split())
                    wf = open(folder_path 
                        + path.split('.')[0].split('/')[-1] + '.pos', 'w')
                    wf.write(str(pos))
                    wf.close()

                    # CFG parser
                    try:
                        parsed = parser.raw_parse(pars_stem)
                    except (TypeError, IndexError, NameError) as e:
                        error_log.append('Unparsable Error:/t' + path + '/n')
                        pass
                    wf = open(folder_path 
                        + path.split('.')[0].split('/')[-1] + '.pars', 'w')
                    s_pars = " ".join(str(x) for x in list(parsed))
                    s_pars = s_pars.replace("Tree", "")
                    s_pars = s_pars.replace("[","")
                    s_pars = s_pars.replace("]","")
                    s_pars = s_pars.replace("\'","")
                    wf.write(s_pars)
                    wf.close()

    # Print files paths with Errors
    if error_log:
        wf = open(folder_path + 'error_log', 'wb')
        for line in error_log:
            wf.write(line)
        wf.close()

def read_norms(file):
    """ (file open for reading) -> dict of {str: list of str}

    Read the contents of file and return a dictionary where each key is a
    word and each value is a list of strings representing AoA, IMG, and FAM.

    """

    norms = {}
    with open(file, 'r') as csvfile:
        myreader = csv.reader(csvfile, delimiter=',')
        next(myreader) # skip header
        for row in myreader:
            norms[ row[1]] = [row[3], row[4], row[5] ] # Get AoA, IMG, FAM

    return norms

# Feature 1
def count_words(stem_file):
    """ (stem file open for reading) -> int

    Return the number of words in stem_file.
    """
    #f = open(stem_file, "r+") #remove after testing

    return len(stem_file.readlines())

# Feature 2
def avg_characters(stem_file):

    #f = open(stem_file, "r+") #remove after testing (stem_file will be sufficient as it will be already opened in extract_features)
    lines = stem_file.readlines()

    sum = 0

    for word in lines:
        if lines.index(word) == len(lines)-1:
            sum += len(word)

        else: #remove the extra "\n" at the end of each line (occurs for all lines except last one)
            sum += len(word)-1

    return sum/len(lines)

# Feature 3
def honore(stem_file):

    #f = open(stem_file,"r+")  # remove after testing (stem_file will be sufficient as it will be already opened in extract_features)

    l = stem_file.readlines()
    for i in range(len(l)):
        l[i] = l[i].strip("\n")

    n = len(l)
    v = len(set(l))

    v1 = 0
    for word in l:
        temp = l[:]
        temp.remove(word)

        if word not in temp:
            v1 += 1

    if v == v1:
        return 0

    else:
        return (100*math.log(n,10)/(1-(v1/v)))

# Feature 4
def tree_depth(pars_file):
    #f = open(pars_file, "r+")  # remove after testing (pars_file will be sufficient as it will be already opened in extract_features)

    l = pars_file.readlines()
    for i in range(len(l)):
        l[i] = l[i].strip("\n")

    max = 0

    for word in l:
        temp = 0

        for char in reversed(word):
            if char == ")":
                temp += 1

            else:
                break

        if temp > max:
            max = temp

    return max

# Feature 5
def cc(pos_file):

    #f = open(pos_file, "r+")  # remove after testing (pos_file will be sufficient as it will be already opened in extract_features)

    l = eval(pos_file.readline())

    sum = 0

    for item in l:
        if "CC" in item:
            sum += 1

    return sum

# Feature 6
def vbg(pos_file):
    #f = open(pos_file,"r+")  # remove after testing (pos_file will be sufficient as it will be already opened in extract_features)

    l = eval(pos_file.readline())

    sum = 0

    for item in l:
        if "VBG" in item:
            sum += 1

    return sum

# Feature 7
def vbz_vbp(pos_file):
    #f = open(pos_file, "r+")  # remove after testing (pos_file will be sufficient as it will be already opened in extract_features)

    l = eval(pos_file.readline())

    sum = 0

    for item in l:
        if "VBZ" in item or "VBP" in item:
            sum += 1

    return sum

# Feature 8
def aoa(stem_file):
    #f = open(stem_file,"r+")  # remove after testing (stem_file will be sufficient as it will be already opened in extract_features)

    os.chdir("/Users/nicolasahar/Desktop/ECs/Archive/2015-2016/Computing for Medicine (Apr 2016)/Phase 3/Seminar 1/Project 1/C4MProject1")
    dict = read_norms("Norms.csv")

    l = stem_file.readlines()
    running_total = 0
    entries_in_norms = 0

    for word in l:
        if word in dict:
            running_total += int(dict[word][0])
            entries_in_norms += 1

    if entries_in_norms > 0:
        return running_total/entries_in_norms

    return 0

# Feature 9
def compute(pos_file):
    #f = open(pos_file, "r+")  # remove after testing (pos_file will be sufficient as it will be already opened in extract_features)

    l = eval(pos_file.readline())

    numerator = 0
    denominator = 0

    for item in l:
        if item[1] == "NN" or item[1] == "NNS" or item[1] == "NNP" or item[1] == "NNPS":
            numerator += 1

        elif item[1] == "PRP" or item[1] == "PRP$":
            denominator += 1

    if denominator == 0:
        return 0

    return numerator/denominator

def extract_features(flist, path):
    """ (file open for reading, str) -> array

    Return an N x 9 array with nine features for each file in flist
    (one filename per line) in directory path.
    """

    norms = read_norms('Norms.csv')
    features = np.array([]) 

    for filename in flist:

        # 0. Count number of words in utterance
        f = open(path + filename + '.stem', 'r')
        f0 = count_words(f)
        f.close()

        # 1. Count average number of characters in utterance
        f = open(path + filename + '.stem', 'r')
        f1 = avg_characters(f)
        f.close()

        # 2. Compute Honore's statistic on utterance
        f = open(path + filename + '.stem', 'r')
        f2 = honore(f)
        f.close()
  
        # 3. Compute the parse tree depth
        f = open(path + filename + '.pars', 'r')
        f3 = tree_depth(f)
        f.close()

        # 4. Count the number of 'CC' instances in parse
        f = open(path + filename + '.pos', 'r')
        f4 = cc(f)
        f.close()

        # 5. Count the number of 'VBG' instances in parse
        f = open(path + filename + '.pos', 'r')
        f5 = vbg(f)
        f.close()

        # 6. Count the number of 'VBZ' and 'VBP' instances in parse
        f = open(path + filename + '.pos', 'r')
        f6 = vbz_vbp(f)
        f.close()

        # 7. Count the average Age of Acquisiton (AoA) of words
        f = open(path + filename + '.stem', 'r')
        f7 = aoa(f)
        f.close()

        # 8. Compute (NN + NNS + NNP + NNPS) / (PRP + PRP$)
        f = open(path + filename + '.pos', 'r')
        f8 = compute(f)
        f.close()

        vector = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8])
        features = np.vstack([features, vector]) if features.size else vector 

    return features

def classify(DD, CD):
    """ (array, array) -> Nonetype

    Report the outcome of classifying feature vectors.  DD and CD are arrays for 
    participants with and without dementia, respectively.
    """

    # do K-fold cross validation
    all_data  = np.concatenate((DD,CD))
    all_class = np.vstack((np.ones((DD.shape[0], 1)), np.zeros((CD.shape[0], 1))))
    N = all_data.shape[0]
    K = 5
    accuracies = np.zeros((K, 1))
    sensitivities = np.zeros((K, 1))
    specificities = np.zeros((K, 1))
    randIndices = np.random.permutation(range(0, N))

    final = {"Accuracy": accuracies, "Sensitivity": sensitivities, "Specificity": specificities}

    means = np.zeros((K, 1))
    variances = np.zeros((K, 1))

    for fold in range(0, K) :
        i_test = randIndices[fold * (N // K) : (fold + 1) * (N // K)]
        i_train = [val for val in randIndices if val not in i_test]
        c_train = all_class[i_train]
        c_test = all_class[i_test]   

        # train the model using features all_data[i_train,:] and classes c_train 
        my_model = svm.SVC(kernel="linear")

        # creating the test and train slices
        A = []
        B = []
        for i in range(len(all_data)):
            if i in i_train:
                A.append(all_data[i])

            elif i in i_test:
                B.append(all_data[i])

        c_train = np.ravel(c_train)
        my_model.fit(A, c_train)

        my_predictions = np.vstack(my_model.predict(B))
        actual = c_test

        # compute the % errors using all_data[i_test,:], c_test, and the trained model
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i, b in zip(my_predictions, actual):
            if i == 1 and b ==1: #TP
                TP += 1

            elif i == 1 and b == 0: #FP
                FP += 1

            elif i == 0 and b == 1: #FN
                FN += 1

            elif i == 0 and b == 0: #TN
                TN += 1

        accuracies[fold] = (TP + TN)/(TP + TN + FP + FN)
        sensitivities[fold] = TP/(TP + FN)
        specificities[fold] = TN/(TN + FP)

    # report result mean and variance, over all folds, of each of accuracy, specificity, and sensitivity
    for key in final:
        print("%s: mean = %s, variance = %s" %(key, np.mean(final[key]), np.var(final[key])))

if __name__ == "__main__":

    # read list of txt2 files
    df = open("./results/Dementia.list", "r")
    cf = open("./results/Controls.list", "r")

    # load list of txt2 files to be parsed
    dlist = load(df)
    clist = load(cf)

    # close the files
    df.close()
    cf.close()

    # do preprocessing. (We've already taken care of this.)
    #preprocess(dlist, "./results/Dementia/")
    #preprocess(clist, "./results/Controls/")
    
    # extract relevant features
    DD = extract_features(dlist, "./Data/Dementia/")
    CD = extract_features(clist, "./Data/Controls/")

    # do the K-fold crossvalidation classification
    classify(DD, CD )

