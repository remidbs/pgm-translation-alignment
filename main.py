import Corpus
import IBM1
import numpy as np

corpus = Corpus.Corpus("corpus.txt")

ibm1 = IBM1.IBM1(corpus)
ibm1.train(10)


print "\nIBM1 perplexity : ",ibm1.get_perplexity(),"\n"

f2e = np.argmax(ibm1.proba_f_knowing_e,axis=1)
print "IBM1 Translations :"
for i in range(len(corpus.french_words)):
    print corpus.french_words[i]," -> ", corpus.english_words[f2e[i]]

