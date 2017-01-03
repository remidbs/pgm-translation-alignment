import Corpus
import numpy as np

import IBM1
import IBM2
import HMM


print("loading the corpus...")
corpus = Corpus.Corpus("eutrans/training", separator="#")
#corpus = Corpus.Corpus("corpus.txt", separator="---")
corpus.print_corpus_description()
print("...done")

#%% Testing IBM1
# print(" ")
# print("*"*50)
# print(" ")
# print("Building IBM1 item...")
# ibm1 = IBM1.IBM1(corpus)
# print("...done")
# print("starting to train IBM1...")
# ibm1_nb_training_step = 10
# imb1perplexityevol = ibm1.train(ibm1_nb_training_step, verbose=True)
# print("...done")
#
# print "\nIBM1 perplexity : ",ibm1.get_perplexity(),"\n"
#
# f2e = np.argmax(ibm1.proba_f_knowing_e,axis=1)
# print "IBM1 Translations :"
# for i in range(len(corpus.french_words)):
#     print corpus.french_words[i], " --> ", corpus.english_words[f2e[i]]

#ibm1.print_viterbi_alignment(0)
#ibm1.print_viterbi_alignment(4)

mode = "slowdecrease"

#%% Testing IBM2
print(" ")
print("*"*50)
print(" ")
ibm2 = IBM2.IBM2(corpus, mode)
ibm2.proba_f_knowing_e = np.load("ibm1_proba_f_knowing_e.npy")
# ibm2.proba_f_knowing_e = ibm1.proba_f_knowing_e
print("starting to train IBM2...")
ibm2_nb_training_step = 10
imb2perplexityevol = ibm2.train(ibm2_nb_training_step,True)
print("...done")
f2e = np.argmax(ibm2.proba_f_knowing_e,axis=1)

# ibm2bis = IBM2.IBM2(corpus,penalization=9.0)
# ibm2bis.train(10,True)
# f2ebis = np.argmax(ibm2bis.proba_f_knowing_e,axis=1)
# # raw_input("Press Enter to continue...")

print "IBM2 Translations :"
for i in range(len(corpus.french_words)):
    print corpus.french_words[i], " --> ", corpus.english_words[f2e[i]],"," # , corpus.english_words[f2ebis[i]]


#ibm2.print_viterbi_alignment(0)
# ibm2bis.print_viterbi_alignment(0)
#ibm2.print_viterbi_alignment(4)
# ibm2bis.print_viterbi_alignment(4)

#%% testing HMM
print(" ")
print("*"*50)
print(" ")
hmm = HMM.HMM(corpus, mode)
hmm.proba_f_knowing_e =np.load("ibm1_proba_f_knowing_e.npy")
#hmm.proba_f_knowing_e = ibm1.proba_f_knowing_e
print("Starting to train HMM...")
hmm_nb_training_step = 10
hmmperplexityevol = hmm.train(hmm_nb_training_step, True)
print("...done")
f2eTer = np.argmax(hmm.proba_f_knowing_e, axis=1)
print(" ")
print("HMM Translations :")
for i in range(len(corpus.french_words)):
    print corpus.french_words[i], " --> ", corpus.english_words[f2eTer[i]]

# a short function to do a nice plot
# meme si indique comme non utilise, seaborn change l allure du plot donc faut le garder!
import matplotlib.pyplot as plt
import seaborn


def plot_perplexity_evol():
    #Y0 = imb1perplexityevol
    ibm1_nb_training_step = 10
    Y0 = np.array([42,28,23,22,21,21,20.7,20.6,20.5,20.5])
    Y1 = np.insert(imb2perplexityevol, 0, Y0[-1])
    Y2 = np.insert(hmmperplexityevol, 0, Y0[-1])
    plt.plot(Y0, label="ibm1 perplexity")
    plt.plot(range(ibm1_nb_training_step-1,ibm1_nb_training_step + ibm2_nb_training_step), Y1, label="ibm2 perplexity")
    plt.plot(range(ibm1_nb_training_step-1,ibm1_nb_training_step + hmm_nb_training_step), Y2, label="hmm perplexity")
    plt.title("Evolution of perplexity: IBM1 pre-training - comparison of IBM2 and HMM")
    plt.legend()
    plt.show()

plot_perplexity_evol()
