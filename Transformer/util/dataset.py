from torch.utils.data import Dataset

class MultiDataset(Dataset):
    def __init__(self, data, language):
        self.data = data
        self.language = language
        self.sentences = self.preprocess()

    def preprocess(self):
        # dataset 안에 길이가 0인 문장이 존재한다.
        sentences = [ (self.language.src_encode(de), self.language.tgt_encode(eng))
                      for de, eng in self.data if len(eng) > 0 and len(de) > 0]

        return sentences

    def src_max_len(self):
        return max([len(src) for src, tgt in self.sentences])

    def tgt_max_len(self):
        return max([len(tgt) for src, tgt in self.sentences])

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)