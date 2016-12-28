import numpy as np

class IBM2:
    def __init__(self,corpus):
        self.Jmax = corpus.Jmax #max length of a french sentence
        self.Imax = corpus.Imax #max length of an english sentence
        self.corpus = corpus
        self.proba_f_knowing_e = np.ones((len(corpus.french_words),len(corpus.english_words)))
    
    def r(self,x):
        y = 1.0-np.abs(x)*1.0/self.Jmax
        return y*(y>=0)
    
    def train(self,n_iterations, verbose=False):
        n_sentences = len(self.corpus.french_sentences)
        #initilization
        self.proba_f_knowing_e = self.proba_f_knowing_e *1.0 / len(self.corpus.english_words)
        for it in range(n_iterations):
            if verbose:
                print "Iteration nb",it
            count = np.zeros((len(self.corpus.french_words),len(self.corpus.english_words)))
            #position
            for s in range(n_sentences):
                f = self.corpus.french_sentences[s]
                J = len(f)
                e = self.corpus.english_sentences[s]
                I = len(e)
                alignment = self.r(((np.array(range(J))+1)*1.0*I/J)[:,np.newaxis] - (np.array(range(I))+1)[np.newaxis,:])
                alignment = alignment/alignment.sum(axis=1)[:,np.newaxis]
                alignment = alignment*self.proba_f_knowing_e[self.corpus.french_sentences[s],:][:,self.corpus.english_sentences[s]]
                alignment = alignment.argmax(axis=1)
                for j in range(alignment.shape[0]):
                    count[f[j],e[alignment[j]]] += 1
            #parameter estimation
            self.proba_f_knowing_e = count/count.sum(axis=1)[:,np.newaxis]
            if verbose:           
                print self.get_perplexity()
        return
        
    def get_perplexity(self,):
        return -1