class Language:
    pad_token_id = 0
    unk_token_id = 1
    sos_token_id = 2
    eos_token_id = 3

    def __init__(self, src_tokenizer, tgt_tokenizer, src_token2id, tgt_token2id, src_id2token, tgt_id2token):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_token2id = src_token2id
        self.tgt_token2id = tgt_token2id

        self.src_id2token = src_id2token
        self.tgt_id2token = tgt_id2token

    def src_encode(self, src_text):
        source_sentence = [self.src_token2id.get(token.text, Language.unk_token_id) for token in
                           self.src_tokenizer.tokenizer(src_text)]
        return source_sentence

    def tgt_encode(self, tgt_text):
        target_sentence = [self.tgt_token2id['<sos>']] \
                          + [self.tgt_token2id.get(token.text, Language.unk_token_id) for token in
                             self.tgt_tokenizer.tokenizer(tgt_text)] \
                          + [self.tgt_token2id['<eos>']]
        return target_sentence

    def src_decode(self, ids):
        sentence = list(map(lambda x: self.src_id2token[x], ids))
        return " ".join(sentence)

    def tgt_decode(self, ids):
        sentence = list(map(lambda x: self.tgt_id2token[x], ids))[1:-1]
        return " ".join(sentence)