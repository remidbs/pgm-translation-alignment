import numpy as np
import math

class IBM1:
    def __init__(self,corpus):
        self.Jmax = corpus.Jmax #max length of a french sentence
        self.Imax = corpus.Imax #max length of an english sentence
        self.corpus = corpus
        self.proba_J_knowing_I = np.zeros((self.Jmax+1, self.Imax+1)) #coefficient [j,i] contains P(j|i)
        self.proba_f_knowing_e = np.zeros((len(corpus.french_words),len(corpus.english_words)))
    
    def train(self,n_iterations):
        
        n_sentences = len(self.corpus.english_sentences)
        n_french_words = len(self.corpus.french_words)
        n_english_words = len(self.corpus.english_words)

        # Train proba_J_knowing_I 
        
        for s in range(n_sentences):
            j = len(self.corpus.french_sentences[s])
            i = len(self.corpus.english_sentences[s])
            self.proba_J_knowing_I[j,i]+=1

        for i in range(self.Imax):
            self.proba_J_knowing_I[:,i]=self.proba_J_knowing_I[:,i]/max(1,sum(self.proba_J_knowing_I[:,i]))


        # Train proba_f_knowing_e 
        # pre compute sum(delta(f,f_js)) and sum(delta(e,e_is))
        sum_delta_f = np.zeros((n_french_words,n_sentences))
        sum_delta_e = np.zeros((n_english_words,n_sentences))
        for s in range(n_sentences):
            for f in range(len(self.corpus.french_words)):
                sum_delta_f[f,s]=self.corpus.french_sentences[s].count(f)
            for e in range(len(self.corpus.english_words)):
                sum_delta_e[e,s]=self.corpus.english_sentences[s].count(e)
        
        #initialize with uniform translation probabilities
        self.proba_f_knowing_e = np.ones((n_french_words,n_english_words))/n_french_words

        #iterative equation
        for it in range(n_iterations):
            for e in range(n_english_words):
                for f in range(n_french_words):
                    coeff=0
                    for s in range(n_sentences):
                        temp=0
                        for e_is in self.corpus.english_sentences[s]:
                            temp+=self.proba_f_knowing_e[f,e_is]
                        coeff+=sum_delta_f[f,s]*sum_delta_e[e,s]/temp
                        
                    self.proba_f_knowing_e[f,e] = coeff*self.proba_f_knowing_e[f,e]
                #normalize each row
                self.proba_f_knowing_e[:,e]=self.proba_f_knowing_e[:,e]/max(1,sum(self.proba_f_knowing_e[:,e]))
        return

    def get_perplexity(self,):
        n_sentences = len(self.corpus.english_sentences)
        perplexity = 1
        for s in range(n_sentences):
            J = len(self.corpus.french_sentences[s])
            I = len(self.corpus.english_sentences[s])
            for j in range(J):
                temp = 0
                for i in range(I):
                    f = self.corpus.french_sentences[s][j]
                    e = self.corpus.english_sentences[s][i]
                    temp += self.proba_f_knowing_e[f,e]
                perplexity = perplexity * temp
                
            perplexity = perplexity * self.proba_J_knowing_I[J,I] / I
        return 1/math.pow(perplexity, 1.0/n_sentences)