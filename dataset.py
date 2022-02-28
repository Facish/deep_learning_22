import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# based on: https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb#scrollTo=H_Fzhe5qFV3J

class TsvDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        self.file_path = os.path.join(data_dir, type_path)
        
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, title, body, genre_id):
        input = f"{title} {body}"
        target = f"{genre_id}"
        return input, target
  
    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                assert len(line) == 3
                assert len(line[0]) > 0
                assert len(line[1]) > 0
                assert len(line[2]) > 0

                title = line[0]
                body = line[1]
                genre_id = line[2]

                input, target = self._make_record(title, body, genre_id)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )

                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)


if __name__ == '__main__':
    # トークナイザー（SentencePiece）モデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    # テストデータセットの読み込み
    train_dataset = TsvDataset(tokenizer, ".\dataset", "train.tsv", 
                            input_max_len=512, target_max_len=4)

    for data in train_dataset:
        print("A. 入力データの元になる文字列")
        print(tokenizer.decode(data["source_ids"]))
        print()
        print("B. 入力データ（Aの文字列がトークナイズされたトークンID列）")
        print(data["source_ids"])
        print()
        print("C. 出力データの元になる文字列")
        print(tokenizer.decode(data["target_ids"]))
        print()
        print("D. 出力データ（Cの文字列がトークナイズされたトークンID列）")
        print(data["target_ids"])
        break