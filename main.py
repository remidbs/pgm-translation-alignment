import Corpus
import numpy as np

import IBM1
import IBM2
import HMM


print("loading the corpus...")
corpus = Corpus.Corpus("corpus.txt")
corpus.print_corpus_description()
print("...done")

#%% Testing IBM1

print("Building IBM1 item...")
ibm1 = IBM1.IBM1(corpus)
print("...done")
print("starting to train IBM1...")
ibm1.train(10, verbose=True)
print("...done")

print "\nIBM1 perplexity : ",ibm1.get_perplexity(),"\n"

f2e = np.argmax(ibm1.proba_f_knowing_e,axis=1)
print "IBM1 Translations :"
for i in range(len(corpus.french_words)):
    print corpus.french_words[i], " --> ", corpus.english_words[f2e[i]]

ibm1.print_viterbi_alignment(0)
ibm1.print_viterbi_alignment(4)

#%% Testing IBM2


ibm2 = IBM2.IBM2(corpus)
print("starting to train IBM2...")
ibm2.train(10,True)
print("...done")
f2e = np.argmax(ibm2.proba_f_knowing_e,axis=1)

ibm2bis = IBM2.IBM2(corpus,penalization=9.0)
ibm2bis.train(10,True)
f2ebis = np.argmax(ibm2bis.proba_f_knowing_e,axis=1)

print "IBM2 Translations :"
for i in range(len(corpus.french_words)):
    print corpus.french_words[i], " --> ", corpus.english_words[f2e[i]],",", corpus.english_words[f2ebis[i]]

ibm2.print_viterbi_alignment(0)
ibm2bis.print_viterbi_alignment(0)
ibm2.print_viterbi_alignment(4)
ibm2bis.print_viterbi_alignment(4)

#%% testing HMM
print(" ")
print(" ***** ")
print(" ")
hmm = HMM.HMM(corpus)
print("Starting to train HMM...")
hmm.train(10,True)
print("...done")
f2eTer = np.argmax(hmm.proba_f_knowing_e, axis = 1)
print("HMM Translations :")
for i in range(len(corpus.french_words)):
    print corpus.french_words[i], " --> ", corpus.english_words[f2eTer[i]]
