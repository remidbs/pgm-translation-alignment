import numpy as np

class IBM2:
    def __init__(self,corpus, penalization = 0.0):
        self.Jmax = corpus.Jmax #max length of a french sentence
        self.Imax = corpus.Imax #max length of an english sentence
        self.corpus = corpus
        self.proba_f_knowing_e = np.ones((len(corpus.french_words),len(corpus.english_words))) *1.0 / len(self.corpus.english_words)
        self.loglikelihood = 0.0
        self.penalization = penalization
    
    #r computes the unnormalized probability of an alignment p(i|j,J,I)
    #the penalization argument must be set low to get uniform alignement probabilities
    def r(self,x):
        y = 1.0*self.Jmax-np.abs(x)-self.penalization
        return y*(y>=0)
    
    def train(self,n_iterations, verbose=False):
        n_sentences = len(self.corpus.french_sentences)
        for it in range(n_iterations):
            self.loglikelihood = 0.0
            count = np.zeros((len(self.corpus.french_words),len(self.corpus.english_words)))
            #position
            for s in range(n_sentences):
                f = self.corpus.french_sentences[s]
                J = len(f)
                e = self.corpus.english_sentences[s]
                I = len(e)
                
                alignment_probabilities = self.r((np.arange(J)*1.0*I/J)[:,np.newaxis] - np.arange(I)[np.newaxis,:])
                alignment_probabilities /= alignment_probabilities.sum(axis=1)[:,np.newaxis]#normalizing proba
                
                alignment_probabilities *= self.proba_f_knowing_e[f,:][:,e]
                most_likely_alignment = alignment_probabilities.argmax(axis=1)
                self.loglikelihood += np.sum(np.log(alignment_probabilities.max(axis=1)))
                for j in range(most_likely_alignment.shape[0]):
                    count[f[j],e[most_likely_alignment[j]]] += 1
            #parameter estimation
            self.proba_f_knowing_e = count/count.sum(axis=1)[:,np.newaxis]
            if verbose:           
                print "Iteration nb",it,". Perplexity :",self.get_perplexity()
        return
        
    def get_perplexity(self,):
        return np.exp(-self.loglikelihood/len(self.corpus.french_sentences))
        
    def get_viterbi_alignment(self,sentence_index = 0):
        f = self.corpus.french_sentences[sentence_index]
        J = len(f)
        e = self.corpus.english_sentences[sentence_index]
        I = len(e)
        
        alignment_probabilities = self.r((np.arange(J)*1.0*I/J)[:,np.newaxis] - np.arange(I)[np.newaxis,:])
        alignment_probabilities /= alignment_probabilities.sum(axis=1)[:,np.newaxis]#normalizing proba
        
        alignment_probabilities *= self.proba_f_knowing_e[f,:][:,e]
        most_likely_alignment = alignment_probabilities.argmax(axis=1)
        return most_likely_alignment
        
    def print_viterbi_alignment(self, sentence_index = 0):
        self.corpus.print_alignment(sentence_index, self.get_viterbi_alignment(sentence_index))
                
        
        
        
        