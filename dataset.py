import torch
from datasets import load_dataset
import spacy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy.cli

class VocabDict(dict):
    def get_itos(self):
        return {v: k for k, v in self.items()}

class Multi30kDataset:
    def __init__(self, split='train', src_vocab=None, tgt_vocab= None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        print(f"Loading {split} dataset...")
        temp = "bentrevett/multi30k"
        args = (temp, )
        kwargs = {"split" : split}
        self.dataset = load_dataset(*args, **kwargs)
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading missing spacy models...")
            spacy.cli.download('de_core_news_sm')
            spacy.cli.download('en_core_web_sm')
            self.spacy_de = spacy.load('de_core_news_sm')
            self.spacy_en = spacy.load('en_core_web_sm')
        
        self.special_tokens = {
            '<unk>' : 0, '<pad>' : 1, "<sos>" : 2, "<eos>" : 3,
        }
        self.UNK_IDX = self.special_tokens['<unk>']
        self.PAD_IDX = self.special_tokens['<pad>']
        self.SOS_IDX = self.special_tokens['<sos>']
        self.EOS_IDX = self.special_tokens['<eos>']

        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.data = []
        if split == 'train':
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()
    
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        print("Building Vocab")
        src_freqs = {}
        tgt_freqs = {}
        tokenize = lambda model, text: [t.text.lower() for t in model.tokenizer(text)]

        for example in self.dataset:
            for word in tokenize(self.spacy_de, example['de']):
                temp = src_freqs
                temp = temp.get(word, 0)
                temp += 1
                src_freqs[word] = temp
            for word in tokenize(self.spacy_en, example['en']):
                temp = tgt_freqs
                temp = temp.get(word, 0)
                temp+= 1
                tgt_freqs[word] = temp
        src_vocab = VocabDict(self.special_tokens)
        tgt_vocab = VocabDict(self.special_tokens)

        for word, freq in src_freqs.items():
            freq >= 2 and src_vocab.update({word : len(src_vocab)})
        
        for word, freq in tgt_freqs.items():
            freq >= 2 and tgt_vocab.update({word : len(tgt_vocab)})
        
        tgt_vocab.get_itos = lambda: {v: k for k, v in tgt_vocab.items()}
        return src_vocab, tgt_vocab

    def _encode(self, tokens, vocab):
        a = [self.SOS_IDX]
        b = [vocab.get(w, self.UNK_IDX) for w in tokens]
        c = [self.EOS_IDX]
        return a + b + c
    
    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        print("Processing data")
        res = []
        tokenize = lambda model, text: [t.text.lower() for t in model.tokenizer(text)]
        for example in self.dataset:
            args = (self.spacy_de, example['de'])
            src_tokens = tokenize(*args)
            args = (self.spacy_en, example['en'])
            tgt_tokens = tokenize(*args)
            args1 = (src_tokens, self.src_vocab)
            args2 = (tgt_tokens, self.tgt_vocab)

            res.append({
                'src': torch.tensor(self._encode(*args1), dtype = torch.long),
                'tgt': torch.tensor(self._encode(*args2), dtype = torch.long)

            })
        return res
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def collate_fn(batch):
    """
    Pads sentences in a batch so they are all the same length.
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Pad sequences with the PAD_IDX (which is 1)
    src_padded = pad_sequence(src_batch, padding_value=1, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    
    return {'src': src_padded, 'tgt': tgt_padded}