import numpy as np


class Corpus:
    def __init__(self, filepath):
        self.Jmax = -1  # max length of a french sentence
        self.Imax = -1  # max length of an english sentence
        self.french_words = set()
        self.english_words = set()
        self.french_sentences = list()
        self.english_sentences = list()
        
        corpus = open(filepath)
        for line in corpus:
            F, E = line.split("---")
            F = F.split()
            E = E.split()
            for f in F:
                self.french_words.add(f)
            for e in E:
                self.english_words.add(e)
            self.french_sentences.append(F)
            self.english_sentences.append(E)
        # each sentence must have exactly one translation
        assert len(self.french_sentences) == len(self.english_sentences)

        self.Jmax = max([len(F) for F in self.french_words])
        self.Imax = max([len(E) for E in self.english_words])
        self.french_words = np.array(list(self.french_words))
        self.english_words = np.array(list(self.english_words))
        for s in range(len(self.english_sentences)):
            for i in range(len(self.english_sentences[s])):
                self.english_sentences[s][i] =\
                    np.where(self.english_words == self.english_sentences[s][i])[0][0]
            for j in range(len(self.french_sentences[s])):
                self.french_sentences[s][j] =\
                    np.where(self.french_words == self.french_sentences[s][j])[0][0]


    # Just for some comfort

    def corpus_description(self):
        return {
            "number of french words :":len(self.french_words),
            "number of english words :": len(self.english_words),
            "number of french sentences :": len(self.french_sentences),
            "number of english sentences :": len(self.english_sentences),
            "maximal length of a french description :": self.Jmax,
            "maximal length of an english description :": self.Imax
        }

    def print_corpus_description(self):
        for (key, value) in self.corpus_description().items():
            print(key, value)

#################################
###  Minimal working example  ###
#################################
### Uncomment to test

# c = Corpus('corpus.txt')
# for frenchWord in c.french_words:
#     print(frenchWord)
# c.print_corpus_description()
