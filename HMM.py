import numpy as np
import time


class HMM:
    def __init__(self,corpus, mode):
        self.Jmax = corpus.Jmax  # max length of a french sentence
        self.Imax = corpus.Imax  # max length of an english sentence
        self.corpus = corpus
        self.proba_J_knowing_I = np.zeros((self.Jmax+1, self.Imax+1)) # coefficient [j,i] contains P(j|i)
        self.most_likely_alignment = [None]*(len(self.corpus.french_sentences))
        self.scoefs = np.ones((self.Imax*2-1))
        self.perplexity_evolution = []
        self.nb_iterations = 0

        # Rem : A good idea is to use the value returned by IBM1 to have a good init.
        # Tested : it seems to really improve the results !
        self.proba_f_knowing_e = np.ones((len(corpus.french_words),len(corpus.english_words))) * 1. / len(self.corpus.english_words)

        self.mode = mode

    def sfunction(self, x):
        # parameter function to compute penalty etc.
        if self.mode == "slowdecrease":
            return 1./(1. + 2* np.abs(x-1))
        elif self.mode == "gaussian":
            return np.exp(-(x-1)*(x-1) / (2. * 9))/np.sqrt(2 * np.pi)
        elif self.mode == "scoefs":
            return self.scoefs[x+self.Imax-1]
        else:
            print("non valid mode for the s function")

    def train(self,n_iterations, verbose = False):
        # Step 0 : define useful constants
        n_sentences = len(self.corpus.english_sentences)
        n_french_words = len(self.corpus.french_words)
        n_english_words = len(self.corpus.english_words)

        # Train proba_J_knowing_I
        for s in range(n_sentences):
            j = len(self.corpus.french_sentences[s])
            i = len(self.corpus.english_sentences[s])
            self.proba_J_knowing_I[j,i] += 1
        # normalization (paying attention to columns not encountered to avoid dividing by zero)
        self.proba_J_knowing_I /= np.vectorize(lambda x : (x==0)*1 + x)(self.proba_J_knowing_I.sum(axis=0))[np.newaxis,:]


        # Train proba_f_knowing_e
        for it in range(n_iterations):
            t0 = time.clock()
            # We want to alternate parameter estimation and alignment finding
            # For given parameters, the optimal alignment is computed with Q(i,j)
            # For given optimal alignment, the parameters p(f|e) are derived from MLE.

            # Count matrix is storing occurrence of words
            count = np.zeros((n_french_words, n_english_words))
            newcoefs = np.ones((self.Imax*2-1))

            # We iterate and cross improve the estimation over all sentences
            for s in range(n_sentences):
                # First step : we build p(i,i' | I)
                f = self.corpus.french_sentences[s]
                J = len(f)
                e = self.corpus.english_sentences[s]
                I = len(e)

                alignment_probabilities = np.array([[self.sfunction(i1 - i2) for i2 in range(I)] for i1 in range(I)])
                alignment_probabilities /= alignment_probabilities.sum(axis=1)[:,np.newaxis]

                Q = np.ones((I,J))

                # **** Fill Q with the corresponding dynamic recursion formula, using current estimation of p(f|e)
                # Return an estimation of best corresponding alignment
                # Rem : first column of Q is assuming Q(before) = 1
                #Q[:,0]= np.array([self.proba_f_knowing_e[f[0],e[i]] * np.max(np.array([alignment_probabilities[i,i2] for i2 in range(I)])) for i in range(I)])
                Q[:,0]= np.array([self.proba_f_knowing_e[f[0],e[i]]  for i in range(I)])
                for j in range(1,J):
                    for i in range(I):
                        Q[i,j] = self.proba_f_knowing_e[f[j],e[i]] * np.max(np.array([alignment_probabilities[i,i2] * Q[i2, j-1] for i2 in range(I)]))
                
                self.most_likely_alignment[s] = np.array([np.argmax(Q[:,j]) for j in range(J)])
                
                #Update scoefs
                for j in range(J-1):
                    newcoefs[self.most_likely_alignment[s][j+1]-self.most_likely_alignment[s][j]+self.Imax-1] += 1

                # Now we can easily derive from MLE the updated expression of p(f|e) maximizing p(f^J | e^I)
                for j in range(J):
                    count[f[j],e[self.most_likely_alignment[s][j]]] += 1

            self.proba_f_knowing_e = count/count.sum(axis=1)[:,np.newaxis]
            self.scoefs = newcoefs / self.corpus.normalization_for_hmm           
            
            perplexity, alignment_perplexity, translation_perplexity = self.get_perplexity()
            self.perplexity_evolution += [perplexity]
            self.nb_iterations += 1

            if verbose:
                print "Iteration nb",it,". Perplexity :",perplexity,". Alignment perplexity :",alignment_perplexity,". Translation perplexity :",translation_perplexity,"(",time.clock()-t0," sec)"
        
    def get_perplexity(self,):
        alignment_loglikelihood = 0.0
        translation_loglikelihood = 0.0
        normalization_loglikelihood = 0.0
        for s in range(len(self.corpus.french_sentences)):
            f = self.corpus.french_sentences[s]
            J = len(f)
            e = self.corpus.english_sentences[s]
            I = len(e)
            alignment_probabilities = np.array([[self.sfunction(i1 - i2) for i2 in range(I)] for i1 in range(I)])
            alignment_probabilities /= alignment_probabilities.sum(axis=1)[:,np.newaxis]
            #add alignments likelihood
            alignment_loglikelihood += np.sum(np.log([alignment_probabilities[self.most_likely_alignment[s][j],self.most_likely_alignment[s][j-1]] for j in range(J)]))
            #add translation likelihood
            translation_loglikelihood += np.sum(np.log([self.proba_f_knowing_e[f[j],e[self.most_likely_alignment[s][j]]] for j in range(J)]))
            #add normalization likelihood
            normalization_loglikelihood += np.log(self.proba_J_knowing_I[J,I])
        N = np.sum([len(s) for s in self.corpus.french_sentences])
        loglikelihood = translation_loglikelihood + normalization_loglikelihood + alignment_loglikelihood
        return np.exp(-loglikelihood/N),np.exp(-alignment_loglikelihood/N),np.exp(-translation_loglikelihood/N)

    def get_viterbi_alignment(self,sentence_index = 0):
        return self.most_likely_alignment[sentence_index]
        
    def print_viterbi_alignment(self, sentence_index = 0):
        self.corpus.print_alignment(sentence_index, self.get_viterbi_alignment(sentence_index))



