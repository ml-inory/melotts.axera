from tokenizers import Tokenizer

class FastTokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
    def tokenize(self, text):
        """完全模拟 AutoTokenizer.tokenize() 的行为"""
        return self.tokenizer.encode(text).tokens[1:-1]  # 去掉[CLS]和[SEP]