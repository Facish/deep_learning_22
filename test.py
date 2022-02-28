import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import textwrap
import argparse
from tqdm.auto import tqdm
from sklearn import metrics

from dataset import TsvDataset

# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(f"./model/", is_fast=True)

# 学習済みモデル
trained_model = AutoModelForSeq2SeqLM.from_pretrained(f"./model/")

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()

# 各種ハイパーパラメータ
args_dict = dict(
    data_dir="/content/data",  # データセットのディレクトリ
    model_name_or_path="rinna/japanese-gpt-1b",
    tokenizer_name_or_path="rinna/japanese-gpt-1b",

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    # max_input_length=512,
    # max_target_length=4,
    # train_batch_size=8,
    # eval_batch_size=8,
    # num_train_epochs=4,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)
args_dict.update({
    "max_input_length":  512,  # 入力文の最大トークン数
    "max_target_length": 4,  # 出力文の最大トークン数
    "train_batch_size":  8,
    "eval_batch_size":   8,
    "num_train_epochs":  1,
    })
args = argparse.Namespace(**args_dict)

# テストデータの読み込み
test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv", 
                          input_max_len=args.max_input_length, 
                          target_max_len=args.max_target_length)

test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

trained_model.eval()

outputs = []
confidences = []
targets = []

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    outs = trained_model.generate(input_ids=input_ids, 
        attention_mask=input_mask, 
        max_length=args.max_target_length,
        return_dict_in_generate=True,
        output_scores=True)

    dec = [tokenizer.decode(ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False) 
                for ids in outs.sequences]
    conf = [s.cpu().item() for s in torch.exp(outs.sequences_scores)]
    target = [tokenizer.decode(ids, skip_special_tokens=True, 
                               clean_up_tokenization_spaces=False) 
                for ids in batch["target_ids"]]

    outputs.extend(dec)
    confidences.extend(conf)
    targets.extend(target)

metrics.accuracy_score(targets, outputs)
print(metrics.classification_report(targets, outputs))