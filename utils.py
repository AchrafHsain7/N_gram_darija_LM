from collections import defaultdict
import csv
import numpy as np 


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
            for i in range(1,len(sent) - 2):
                patterns[(sent[i], sent[i+1])] += freq
            # print("".join(sentence).count(pattern))
        try:
            self.frequent_pattern =  max(patterns, key=patterns.get)
        except:
            print(">>>MAX Word Tokenization Reached")
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


    def get_vocab(self):
        return list(self.vocab)


    def save_vocab(self):
        self.get_unique_tkn()
        with open("vocab.txt", "w", encoding="utf-8") as f:
            f.writelines([x + "\n" for x in self.vocab])

    def load_vocab(self, f_path):
        self.vocab = set()
        with open(f_path, "r", encoding="utf-8") as f:
            tkns = f.readlines()
            for tkn in tkns:
                tkn = tkn.replace("\n", "")
                self.vocab.add(tkn)







class NgramModel:
    def __init__(self, tokenizer, n_gram=2):
        self.corpus = self.process_corpus(tokenizer.train_vocab)
        self.vocab = self.encode_vocab(tokenizer.get_vocab()) 
        self.vocab_len = len(self.vocab)
        self.n_gram = n_gram
        self.get_inverse_vocab()

    def process_corpus(self, corpus):
        corpus = [x.split() for x in corpus]
        return corpus
    
    def load_vocab(self, f_path):
        self.vocab = set()
        with open(f_path, "r", encoding="utf-8") as f:
            tkns = f.readlines()
            for tkn in tkns:
                tkn = tkn.replace("\n", "")
                self.vocab.add(tkn)

        self.vocab_len = len(self.vocab)
        self.vocab = self.encode_vocab(list(self.vocab))
        self.get_inverse_vocab()

    
    def encode_vocab(self, vocab):
        encoded_vocab = defaultdict(int)
        for i in range(len(vocab)):
            encoded_vocab[vocab[i]] += i
        
        return encoded_vocab
    
    def get_inverse_vocab(self):
        self.inverse_vocab = {}
        for key, val in self.vocab.items():
            self.inverse_vocab[val] = key

    
    def initialize_model(self):
        #rows: are w1
        #columns: are w2
        #model: probability of w2 given we have w1
        self.model = np.ones((self.vocab_len, self.vocab_len))

    def normalize_model(self):
        sum_prob = np.sum(self.model)
        self.model = self.model / sum_prob

    def train_model(self):
        self.initialize_model()
        for sent in self.corpus:
            for i in range(len(sent)-1):
                # getting w2 given we have w1
                w1 = sent[i]
                w2 = sent[i+1]

                idx1 = self.vocab[w1]
                idx2 = self.vocab[w2]

                #increase probability of w2 after w1
                self.model[idx1][idx2] += 1

        self.normalize_model()

    def generate_text(self, start="<SOS>"):
        response = start 
        current = start
        if start in self.vocab.keys():
            while "<EOS" not in current:
                current_idx = self.vocab[current]
                # adding some randomness for more intersting results
                prob = self.model[current_idx] / np.sum(self.model[current_idx])
                next_idx = np.random.choice(np.arange(self.vocab_len), size=1, p=prob)[0]
                next_tkn = self.inverse_vocab[next_idx]
                response += next_tkn
                current = next_tkn
            print(response)
        else:
            print(start + " Not in Vocabulary")
                


        


    


