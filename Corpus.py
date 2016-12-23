import numpy as np

class Corpus:
    def __init__(self, filepath):
        self.Jmax = -1 #max length of a french sentence
        self.Imax = -1 #max length of an english sentence
        self.french_words = set()
        self.english_words = set()
        self.french_sentences = list()
        self.english_sentences = list()
        
        corpus = open(filepath)
        for line in corpus:
            F,E = line.split("\t")
            F = F.split()
            E = E.split()
            for f in F:
                self.french_words.add(f)
            for e in E:
                self.english_words.add(e)
            self.french_sentences += [F]
            self.english_sentences += [E]
            self.Jmax = max(len(F),self.Jmax)
            self.Imax = max(len(E),self.Imax)
        self.french_words = np.array(list(self.french_words))
        self.english_words = np.array(list(self.english_words))
        for s in range(len(self.english_sentences)):
            for i in range(len(self.english_sentences[s])):
                self.english_sentences[s][i] =\
                    np.where(self.english_words == self.english_sentences[s][i])[0][0]
            for j in range(len(self.french_sentences[s])):
                self.french_sentences[s][j] =\
                    np.where(self.french_words == self.french_sentences[s][j])[0][0]
        
        