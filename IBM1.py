import numpy as np
import math
import time


class IBM1:
    def __init__(self, corpus):
        self.Jmax = corpus.Jmax  # max length of a french sentence
        self.Imax = corpus.Imax  # max length of an english sentence
        self.corpus = corpus
        self.loglikelihood = 0.0
        self.proba_J_knowing_I = np.zeros((self.Jmax+1, self.Imax+1)) # coefficient [j,i] contains P(j|i)
        self.proba_f_knowing_e = np.zeros((len(corpus.french_words),len(corpus.english_words)))
    
    def train(self, n_iterations, verbose=False):
        
        n_sentences = len(self.corpus.english_sentences)
        n_french_words = len(self.corpus.french_words)
        n_english_words = len(self.corpus.english_words)
        perplexity_evolution = np.zeros(n_iterations)

        # Train proba_J_knowing_I 
        for s in range(n_sentences):
            j = len(self.corpus.french_sentences[s])
            i = len(self.corpus.english_sentences[s])
            self.proba_J_knowing_I[j,i] += 1

        for i in range(self.Imax):
            self.proba_J_knowing_I[:,i] = self.proba_J_knowing_I[:,i]/max(1,sum(self.proba_J_knowing_I[:,i]))

        # Train proba_f_knowing_e
        # pre compute sum(delta(f,f_js)) and sum(delta(e,e_is))
        count_f = np.zeros((n_french_words, n_sentences))
        count_e = np.zeros((n_english_words, n_sentences))
        # t0 = time.clock()
        for s in range(n_sentences):
            for f in range(n_french_words):
                count_f[f,s] = self.corpus.french_sentences[s].count(f)
            for e in range(n_english_words):
                count_e[e,s] = self.corpus.english_sentences[s].count(e)
        
        # initialize with uniform translation probabilities
        self.proba_f_knowing_e = np.ones((n_french_words,n_english_words))/n_french_words

        # iterative equation
        for it in range(n_iterations):
            t0 = time.clock()
            A = np.zeros((len(self.corpus.french_words), len(self.corpus.english_words)))
            # t0_bis = time.clock()
            # print "t0_bis - t0", t0_bis - t0
            for s in range(n_sentences):
                t1_ante = time.clock()
                e = self.corpus.english_sentences[s]
                # t1 = time.clock()
                temp1 = np.outer(count_f[:, s], count_e[:, s])
                # t1_bis = time.clock()
                temp2 = np.tile( (lambda x : (x==0)*1 + x)(self.proba_f_knowing_e[:, e].sum(axis=1)), (n_english_words, 1) )
                t1_ter_ante = time.clock()
                A += temp1 / np.transpose(temp2)
                t1_ter = time.clock()

                ### Uncomment if you want to see how much time each iteration takes
                # if (s % 1000) == 0:
                #     print "Calcul A += temp1/ temp2", t1_ter - t1_ter_ante
                #     print "Duree d une etape complete :", t1_ter - t1_ante

            t2 = time.clock()
            self.proba_f_knowing_e *= A
            self.proba_f_knowing_e /= (lambda x : (x==0)*1 + x)(self.proba_f_knowing_e.sum(axis=0)[np.newaxis, :])
            t2_bis = time.clock()
            # print "t2_bis - t2", t2_bis - t2
            perplexity = self.get_perplexity()
            perplexity_evolution[it] = perplexity
            if verbose:
                print "Iteration nb ",it,". Perplexity :",perplexity, " (", time.clock()-t0," sec)"

        return perplexity_evolution

    def get_perplexity(self,):
        self.loglikelihood = 0.0
        n_sentences = len(self.corpus.english_sentences)
        for s in range(n_sentences):
            J = len(self.corpus.french_sentences[s])
            I = len(self.corpus.english_sentences[s])
            f = self.corpus.french_sentences[s]
            e = self.corpus.english_sentences[s]
            self.loglikelihood += np.sum(np.log(np.sum(self.proba_f_knowing_e[f,:][:,e],axis=1)))
            self.loglikelihood += np.log(self.proba_J_knowing_I[J,I])
            self.loglikelihood -= J*np.log(I)
        return np.exp(-self.loglikelihood/np.sum([len(s) for s in self.corpus.french_sentences]))

    def get_viterbi_alignment(self, sentence_index = 0):
        f = self.corpus.french_sentences[sentence_index]
        e = self.corpus.english_sentences[sentence_index]        
        most_likely_alignment = self.proba_f_knowing_e[f,:][:,e].argmax(axis=1)
        return most_likely_alignment
        
    def print_viterbi_alignment(self, sentence_index = 0):
        self.corpus.print_alignment(sentence_index, self.get_viterbi_alignment(sentence_index))
