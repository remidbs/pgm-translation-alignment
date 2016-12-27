import Corpus
import IBM1
import numpy as np

corpus = Corpus.Corpus("corpus.txt")

ibm1 = IBM1.IBM1(corpus)
ibm1.train(10)


print np.argmax(ibm1.proba_f_knowing_e,axis=1)
print corpus.english_words
print corpus.french_words
print corpus.english_sentences
print corpus.french_sentences