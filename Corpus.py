import numpy as np
import string


class Corpus:
    def __init__(self, filepath, separator ="---"):
        self.Jmax = -1  # max length of a french sentence
        self.Imax = -1  # max length of an english sentence
        self.french_words = set()
        self.english_words = set()
        self.french_sentences = list()
        self.english_sentences = list()

        corpus = open(filepath)
        for line in corpus:
            F, E = line.split(separator)
            F = F.translate(None, string.punctuation).split()
            E = E.translate(None, string.punctuation).split()
            for f in F:
                self.french_words.add(f)
            for e in E:
                self.english_words.add(e)
            self.french_sentences.append(F)
            self.english_sentences.append(E)
        # each sentence must have exactly one translation
        assert len(self.french_sentences) == len(self.english_sentences)

        self.Jmax = max([len(F) for F in self.french_sentences])
        self.Imax = max([len(E) for E in self.english_sentences])
        self.french_words = np.array(list(self.french_words))
        self.english_words = np.array(list(self.english_words))
        for s in range(len(self.english_sentences)):
            for i in range(len(self.english_sentences[s])):
                self.english_sentences[s][i] =\
                    np.where(self.english_words == self.english_sentences[s][i])[0][0]
            for j in range(len(self.french_sentences[s])):
                self.french_sentences[s][j] =\
                    np.where(self.french_words == self.french_sentences[s][j])[0][0]
                    
        self.normalization_for_hmm = 0
        self.compute_normalization_for_hmm()

    #########################
    # Just for some comfort #
    #########################

    def corpus_description(self):
        return {
            "number of french words :":len(self.french_words),
            "number of english words :": len(self.english_words),
            "number of french sentences :": len(self.french_sentences),
            "number of english sentences :": len(self.english_sentences),
            "maximal length of a french sentence :": self.Jmax,
            "maximal length of an english sentence :": self.Imax
        }

    def print_corpus_description(self):
        print("Corpus description:")
        for (key, value) in self.corpus_description().items():
            print(key, value)
            
    def print_alignment(self, sentence_index, alignment):
        f = self.french_sentences[sentence_index]
        J = len(f)
        e = self.english_sentences[sentence_index]
        I = len(e)
        
        max_length_of_english_word = np.max(np.vectorize(len)(self.english_words[e]))
        max_length_of_french_word = np.max(np.vectorize(len)(self.french_words[f]))
        for j in range(J):
            print (self.french_words[f[j]]).rjust(max_length_of_french_word),
            for i in range(I):
                if i == alignment[j]:
                    print "X",
                else:
                    print " ",
            print
        for k in range(max_length_of_english_word):
            print (" ").rjust(max_length_of_french_word),
            for i in range(I):
                if k < len(self.english_words[e[i]]):
                    print self.english_words[e[i]][k],
                else:
                    print " ",
            print 
            
    def compute_normalization_for_hmm(self,):
        self.normalization_for_hmm = np.zeros(self.Imax)
        lengths_sentences = np.array([len(sentence) for sentence in self.english_sentences])
        frequences_of_lengths = np.bincount(lengths_sentences)
        for i in range(self.Imax):
            self.normalization_for_hmm[i] = np.sum(frequences_of_lengths\
                        *(lambda x : x * (x >= 0))(np.array(range(self.Imax+1))-i))
        self.normalization_for_hmm = np.array(list(reversed(self.normalization_for_hmm[1:]))+list(self.normalization_for_hmm))
        self.normalization_for_hmm += (self.normalization_for_hmm == 0)           

#################################
###  Minimal working example  ###
#################################

def minimal_working_example():
    c = Corpus('corpus.txt')
    for frenchWord in c.french_words:
        print(frenchWord)
    c.print_corpus_description()
