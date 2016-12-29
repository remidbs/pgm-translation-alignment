import numpy as np


class HMM:
    def __init__(self,corpus):
        self.Jmax = corpus.Jmax  # max length of a french sentence
        self.Imax = corpus.Imax  # max length of an english sentence
        self.corpus = corpus

        # Rem : A better idea is to use the value returned by IBM1 to have a good init.
        self.proba_f_knowing_e = np.ones((len(corpus.french_words),len(corpus.english_words))) *1.0 / len(self.corpus.english_words)

    def sfunction(self, x):
        # parameter function to compute penalty etc.
        # Arbitrary choice atm, must be improved. Maybe gaussian ?
        return 1./(1. + np.abs(x))

    def train(self,n_iterations, verbose = False):
        # Step 0 : define useful constants
        n_sentences = len(self.corpus.english_sentences)
        n_french_words = len(self.corpus.french_words)
        n_english_words = len(self.corpus.english_words)

        for it in range(n_iterations):
            # We want to alternate parameter estimation and alignment finding
            # For given parameters, the optimal alignment is computed with Q(i,j)
            # For given optimal alignment, the parameters p(f|e) are derived from MLE.

            # Count matrix is storing occurrence of words
            count = np.zeros((n_french_words, n_english_words))

            # We iterate and cross improve the estimation over all sentences
            for s in range(n_sentences):
                # First step : we build p(i,i' | I)
                f = self.corpus.french_sentences[s]
                J = len(f)
                e = self.corpus.english_sentences[s]
                I = len(e)
                alignment_probabilities = np.array([[self.sfunction(i1 - i2) for i2 in range(I)] for i1 in range(I)])
                alignment_probabilities /= alignment_probabilities.sum(axis=1)

                Q = np.ones((I,J))

                # **** Fill Q with the corresponding dynamic recursion formula, using current estimation of p(f|e)
                # Return an estimation of best corresponding alignment
                # Rem : first column of Q is assuming Q(before) = 1
                Q[:,0]= np.array([self.proba_f_knowing_e[f[0],e[i]] * np.max(np.array([alignment_probabilities[i,i2] for i2 in range(I)])) for i in range(I)])
                for j in range(2,J):
                    for i in range(I):
                        Q[i,j] = self.proba_f_knowing_e[f[j],e[i]] * np.max(np.array([alignment_probabilities[i,i2] * Q[i2, j-1] for i2 in range(I)]))

                most_likely_alignment = np.array([np.argmax(Q[:,j]) for j in range(J)])

                # Now we can easily derive from MLE the updated expression of p(f|e) maximizing p(f^J | e^I)
                for j in range(J):
                    count[f[j],e[most_likely_alignment[j]]] += 1

            self.proba_f_knowing_e = count/count.sum(axis=0)
            if verbose:
                print "Iteration nb",it,". Perplexity :",self.get_perplexity()
        return
        
    def get_perplexity(self,):
        return -1
