import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import argparse
from tqdm.auto import tqdm
from sklearn import metrics

from dataset import TsvDataset

# based on: https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb#scrollTo=H_Fzhe5qFV3J

# hyper parameters
args_dict = dict(
    data_dir="./dataset/",  # Directory of datasets
)
args_dict.update({
    "max_input_length":  512,
    "max_target_length": 4,
})
args = argparse.Namespace(**args_dict)


def test():
    tokenizer = T5Tokenizer.from_pretrained(f"./models/", is_fast=True)
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(f"./models/")

    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        trained_model.cuda()

    # read test data
    test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv", 
                            input_max_len=args.max_input_length, 
                            target_max_len=args.max_target_length)

    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    trained_model.eval()

    outputs = []
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
        target = [tokenizer.decode(ids, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False) 
                    for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    metrics.accuracy_score(targets, outputs)
    print(metrics.classification_report(targets, outputs))


if __name__ == '__main__':
    test()
