from collections import defaultdict
import csv


class DarijaBPETokenizer:

    def __init__(self, corpus):
        self.corpus = self.process_data(corpus)
        self.train_vocab = self.create_vocab()
        self.get_unique_tkn()

    def process_data(self, data):
        data = [x.lower().strip() for x in data]
        data = [x.replace("\n", "") for x in data]
        return data


    def create_vocab(self):
        #making sure vocab always start from 0 by default
        vocab = defaultdict(int)
        for sent in self.corpus:
            for word in sent.split():
                vocab["<SOS> " + ' '.join(list(word)) + " <EOS>"] += sent.count(word)
        return vocab


    def pattern_frequencies(self):
        patterns = defaultdict(int)
        for word, freq in self.train_vocab.items():
            sent = word.split()
            for i in range(len(sent) - 1):
                patterns[(sent[i], sent[i+1])] += freq
            # print("".join(sentence).count(pattern))
        try:
            self.frequent_pattern =  max(patterns, key=patterns.get)
        except:
            print(">>>Word Tokenization Reached")
            return 0

        

    def update_vocab(self):
        new_keys = []
        for word in self.train_vocab.keys():
            word = word.split()
            sent = []
            for i in range(0, len(word)):
                if i == 0:
                    sent.append(word[i])
                    continue

                if word[i-1] == self.frequent_pattern[0] and word[i] == self.frequent_pattern[1]:
                    sent.pop()
                    sent.append(self.frequent_pattern[0] + self.frequent_pattern[1])
                    continue

                else:
                    sent.append(word[i])
    

            new_keys.append((" ".join(sent), " ".join(word)))
        
        for key in new_keys:
            self.train_vocab[key[0]] = self.train_vocab.pop(key[1])

    
    def get_unique_tkn(self):
        try:
            self.vocab = set(self.vocab)
        except:
            self.vocab = set()
        for sent in self.train_vocab.keys():
            words = sent.split()
            for w in words:
                self.vocab.add(w)
            


    def bp_encode(self, max_vocab=1000, debug=False):
            k = max_vocab - len(self.vocab)
            self.create_vocab()
            for i in range(k):
                stop = self.pattern_frequencies()
                if stop == 0:
                    return
                self.update_vocab()
            self.get_unique_tkn()


    def save_vocab(self):
        self.get_unique_tkn()
        with open("vocab.csv", "w", encoding="utf-8") as f:
            f.writelines([x + "\n" for x in self.vocab])

    def load_vocab(self):
        pass


    


