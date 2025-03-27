from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from typing import List

class JapaneseTokenizer:
    def __init__(self, tokenizer_dir: str):
        """
        初始化 JapaneseTokenizer。
        
        :param tokenizer_dir: 包含 vocab.txt 的目录路径。
        """
        # 加载 vocab.txt 到 WordPiece 模型
        self.tokenizer = Tokenizer(WordPiece.from_file(f"{tokenizer_dir}/vocab.txt", unk_token="[UNK]"))
        
        # 设置预分词器（按空格分词）
        self.tokenizer.pre_tokenizer = Whitespace()
    
    def tokenize(self, text: str) -> List[str]:
        """
        将输入文本分词为 token 列表。
        
        :param text: 输入文本。
        :return: 分词后的 token 列表。
        """
        output = self.tokenizer.encode(text)
        return output.tokens


# 示例用法
if __name__ == "__main__":
    # 假设模型文件位于 "./cl-tohoku/bert-base-japanese-v3"
    tokenizer = JapaneseTokenizer("./cl-tohoku/bert-base-japanese-v3")
    
    text = "日本語のテキストをトークン化します。"
    tokens = tokenizer.tokenize(text)
    print("Tokens:", tokens)