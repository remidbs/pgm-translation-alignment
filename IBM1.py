import numpy as np

class IBM1:
    def __init__(self,corpus):
        self.Jmax = corpus.Jmax #max length of a french sentence
        self.Imax = corpus.Imax #max length of an english sentence
        self.corpus = corpus
        self.proba_J_knowing_I = np.array((self.Jmax, self.Imax))
    
    def train(self,):
        return
        
    def get_perplexity(self,):
        return -1