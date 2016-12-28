import Corpus
import IBM1
import numpy as np

print("loading the corpus...")
corpus = Corpus.Corpus("corpus.txt")
corpus.print_corpus_description()
print("...done")

print("Building IBM1 item...")
ibm1 = IBM1.IBM1(corpus)
print("...done")
print("starting to train IBM1...")
ibm1.train(10)
print("...done")

print "\nIBM1 perplexity : ",ibm1.get_perplexity(),"\n"

f2e = np.argmax(ibm1.proba_f_knowing_e,axis=1)
print "IBM1 Translations :"
for i in range(len(corpus.french_words)):
    print corpus.french_words[i], " --> ", corpus.english_words[f2e[i]]

