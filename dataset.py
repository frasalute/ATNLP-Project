from torch.utils.data import Dataset 
import torch 

class Vocabulary:
    def __init__(self, data, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]):
        self.data = data
        self.special_tokens = {tok: i for i, tok in enumerate(special_tokens)}
        # i, tok in enumerate (special_tokens) create a list with index (i) and value (tok) which gives us
        # indexing of the tokens so ex: 0-->PAD 
        # we then use this to create a dictionary and so we set it to be tok:i because tok is the key and i is
        # the value
        self.id2tok = self._create_id2tok()
        self.tok2id = {v: k for k, v in self.id2tok.items()} # we turn the keys and values
        self.vocab_size = len(self.tok2id)

    def _create_id2tok(self):
        tokens = list(set(" ".join(self.data).split()))
        # combine special tokens and unique tokens into a single listn where the specials come first
        all_tokens = list(self.special_tokens.keys()) + tokens
        return {i: tok for i, tok in enumerate(all_tokens)} # all enumerated starting from 0-3 for specials and 
        # from 4 on for other tokens

class SCANDataset(Dataset): 
    def __init__(self, file_path, max_len=128): 
        self.file_path = file_path 
        self.max_len = max_len
        self.data = self._load_data()
        self.src_vocab = Vocabulary([d["command"] for d in self.data])
        self.tgt_vocab = Vocabulary([d["action"] for d in self.data])

    def _load_data(self): 
        data = [] 
        with open(self.file_path, 'r') as file: 
            for line in file: 
                line = line.strip() 
                if line.startswith("IN:") and "OUT:" in line: 
                    input = line.split("IN:")[1].split("OUT:")[0].strip() 
                    output = line.split("OUT:")[1].strip() 
                    data.append({"command": input, "action": output}) 
        return data

    def encode(self, text, vocab):
        tokens = text.split()
        tokens = [vocab.tok2id.get(tok, vocab.tok2id["<UNK>"]) for tok in tokens] # recognizing a word or unknown
        tokens = [vocab.tok2id["<BOS>"]] + tokens + [vocab.tok2id["<EOS>"]]
        tokens = tokens[:self.max_len-1]
        tokens += [vocab.tok2id["<PAD>"]] * (self.max_len - len(tokens)) # check difference between max lenght and
        # token and adding PADs
        return tokens
    
    def decode(self, tokens, vocab):
        tokens = [int(tok) for tok in tokens]
        tokens = [vocab.id2tok.get(tok, "<UNK>") for tok in tokens]
        tokens = [tok for tok in tokens if tok not in ["<BOS>", "<EOS>", "<PAD>"]]
        return " ".join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        command_text = self.data[idx]["command"]
        action_text = self.data[idx]["action"]
        command_tokens = self.encode(command_text, self.src_vocab) # compare the text to the vocabulary
        action_tokens = self.encode(action_text, self.tgt_vocab)
        return {"command": torch.tensor(command_tokens), "action": torch.tensor(action_tokens)}
        # returning a tensor so we can train the transformer
        
if __name__ == "__main__": 
    dataset = SCANDataset("tasks.txt")
    print(dataset[0])
    print(dataset[0]["command"])
    print(dataset[0]["action"])
    print(dataset.src_vocab.vocab_size)
    print(dataset.tgt_vocab.vocab_size)
    print(dataset.decode(dataset[0]["command"], dataset.src_vocab))
    print(dataset.decode(dataset[0]["action"], dataset.tgt_vocab))


import os
print (os.getcwd())