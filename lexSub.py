import operator 
import numpy as np
import scipy.spatial
import subprocess
import string 
from  collections import OrderedDict
import sys
import kenlm
import en

contentWord = ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'NP', 'RB', 'RBR', 'RBS','VA', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ','VP']

def context(origin, candidate):
    sum1 = 0
    iNum = int(wordNum[origin])
    origin_sentence = wordSentence[origin]
    sentence = ' '.join(origin_sentence)
    tt = len(origin_sentence)
    idx_1 = iNum - 1
    idx1 = iNum + 1
    csim_10 = 1
    csim10 = 1
    
    listTuples = en.sentence.tag(sentence)
    while (idx_1 >=0): 
        word_1 = listTuples[idx_1][0]
        wordPos = listTuples[idx_1][1]
        if ((word_1 in vectorsWords.keys()) and (wordPos in contentWord)):
           csim_10 = 1-scipy.spatial.distance.cosine(vectorsWords[candidate], vectorsWords[word_1]) 
           break
        idx_1 = idx_1 - 1
    while (idx1<tt): 
        word1 = listTuples[idx1][0]
        wordPos = listTuples[idx1][1]
        if ((word1 in vectorsWords.keys()) and (wordPos in contentWord)):
           csim10 = 1-scipy.spatial.distance.cosine(vectorsWords[candidate], vectorsWords[word1]) 
           break
        idx1 = idx1 + 1
    
    sum1 = (csim_10 + csim10)/2.0
    return sum1 


def language(origin, candidate):    
    iddNum = int(wordNum[origin])
    languageSentence = wordSentence[origin]
    listSentence = []
    for val, key in enumerate(languageSentence):
        if val != iddNum:
           listSentence.append(key)
        else:
           listSentence.append(candidate)
    likeliSentence = ' '.join(listSentence)
    model = kenlm.Model('lm-merged.kenlm')
    languageFeatures = model.score(likeliSentence, bos = True, eos = True)
    return languageFeatures

def rankName(x):    
    lists = []
    dictions = {}
    rank = 0
    for candi in x.keys():
        if x[candi]  in lists:
            dictions[candi] = rank
        else:
            rank = rank + 1
            dictions[candi] = rank 
            lists.append(x[candi])
    return dictions

def Simplify():
    bestWords = {}
    for keyy in sorted(wordIdx.iterkeys()):
        scoresSemantic = {}
        scoresContext = {}
        scoresic = {}
        scoresLanguage = {}
        for sc in wordCandidate[int(keyy)]:
            if sc in vectorsWords.keys():
                scoresContext[sc] = context(keyy, sc)
            else:
                scoresContext[sc] = 0
            if sc in icFreq.keys():
                scoresic[sc] = icFreq[sc]
            else:
                scoresic[sc] = 0
            scoresLanguage[sc] = language(keyy,sc)


        maxContext = max(scoresContext.iteritems(), key=operator.itemgetter(1))[1]
        minContext = min(scoresContext.iteritems(), key=operator.itemgetter(1))[1]
        
        maxic = max(scoresic.iteritems(), key=operator.itemgetter(1))[1]
        minic = min(scoresic.iteritems(), key=operator.itemgetter(1))[1]
       
        maxLanguage = max(scoresLanguage.iteritems(), key=operator.itemgetter(1))[1]
        minLanguage = min(scoresLanguage.iteritems(), key=operator.itemgetter(1))[1]
        
        averageScore = {}
        for sc in wordCandidate[int(keyy)]:    
            if maxContext != minContext:
               scoresContext[sc] = (scoresContext[sc] - minContext)/(maxContext - minContext)
            else:
               scoresContext[sc] = 0.5
            if maxic != minic:
               scoresic[sc] = (scoresic[sc] - minic)/(maxic-minic)
            else:
               scoresic[sc] = 0.5
            if maxLanguage != minLanguage:
               scoresLanguage[sc] = (scoresLanguage[sc]-minLanguage)/(maxLanguage-minLanguage)
            else:
               scoresLanguage[sc] = 0.5
            averageScore[sc] = (scoresContext[sc] + scoresic[sc] + scoresLanguage[sc])/3.0
        rAverageScore = sorted(averageScore.items(),key=operator.itemgetter(1), reverse = True)
        rankWord = rankName(OrderedDict(rAverageScore))
               
        
        middleDic = {}
        ss = []
        for key5 in rankWord.keys():
            rankScore = rankWord[key5]
            if rankScore in ss:
                middlelist = []
                if type(middleDic[rankScore]) is list:
                    for dd, vue in enumerate(middleDic[rankScore]):
                        middlelist.append(vue)
                    middlelist.append(key5)
                    middleDic[rankScore] = middlelist
                else:
                    middlelist.append(middleDic[rankScore])
                    middlelist.append(key5)
                    middleDic[rankScore] = middlelist
            else:
                middleDic[rankScore] = key5
                ss.append(rankScore)

        listAll = []
        middle = sorted(middleDic.items(), key=operator.itemgetter(0))
        orderdict = OrderedDict(middle)
        for key5 in orderdict.keys(): 
            listAll.append(orderdict[key5])
        bestWords[int(keyy)] = listAll
    return bestWords    
        


if __name__ == "__main__":
    vectorsWords = {}
    wordNum = {}
    wordIdx = {}
    wordSentence = {}
    wordCandidate = {}
    frelist = []
    repeatlist = []
    icFreq = {}

    with open("glove.6B.200d.txt", "r") as f:
        for x in f.readlines():
            key = x.rstrip().split()[0]
            listVector  = x.rstrip().split()[1:]
            vectorsWords[key] = np.asarray(listVector, dtype = float)

    with open("lst_test.preprocessed", "r") as f:
        for x in f.readlines():
            strWord = x.rstrip().split('\t')[0]
            key = strWord.split('.')[0]
            idx = x.rstrip().split('\t')[1]
            num = x.rstrip().split('\t')[2]
            sentence = x.rstrip().split('\t')[3]
            wordIdx[idx] = key
            wordNum[idx] = num
            sentences = sentence.split(' ')
            wordSentence[idx] = sentences

    with open("substitutions", "r") as f:
        lineNum = 300
        for i,x in enumerate(f.readlines()):
           lineNum = lineNum + 1
           wordCandidate[lineNum] = x.rstrip().split(';')[0:-1]
    with open("fuck" , "r") as f:       
        for x in f.readlines():
            word = x.rstrip().split("\t")[0]
            freq = x.rstrip().split("\t")[1]
            if word not in icFreq.keys():
                 icFreq[word] = float(freq)
           
    rankKeyWords = Simplify()

    with open("caonima", "a") as f:
        for idxx in sorted(rankKeyWords.iterkeys()):
            f.write("Sentence " + str(idxx) + " rankings: ") 
            for idd, value in enumerate(rankKeyWords[idxx]):
                if type(value) is list:
                    listss = value
                    f.write("{")
                    tt = len(listss)
                    for idxxx, valueee in enumerate(listss):
                        if idxxx <  (tt-1):
                            f.write(str(valueee) + ", ")
                        if idxxx == (tt-1):
                            if idd != len(rankKeyWords[idxx])-1:
                                f.write(str(valueee) + "} ")
                            else:
                                f.write(str(valueee) + "}")
                else:
                    if idd != len(rankKeyWords[idxx])-1:
                        f.write("{" + str(value) + "} ")
                    else:
                        f.write("{" + str(value) + "}")
            f.write("\n")
