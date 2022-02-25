import tarfile
import re
import random
from tqdm import tqdm
from normalize_neology import normalize_neologd

# based on: https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb#scrollTo=H_Fzhe5qFV3J

livedoor_dir = "./dataset/ldcc-20140209.tar.gz"
target_genres = ["dokujo-tsushin",
                 "it-life-hack",
                 "kaden-channel",
                 "livedoor-homme",
                 "movie-enter",
                 "peachy",
                 "smax",
                 "sports-watch",
                 "topic-news"]
spacial_tokens_map = {'bos_token': '<s>',
                      'cls_token': '[CLS]',
                      'eos_token': '</s>',
                      'mask_token': '[MASK]',
                      'pad_token': '[PAD]',
                      'sep_token': '[SEP]',
                      'unk_token': '<unk>'}


def remove_brackets(text):
    text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
    return text

def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    return text

def read_title_body(file):
    next(file)
    next(file)
    title = next(file).decode("utf-8").strip()
    title = normalize_text(remove_brackets(title))
    body = normalize_text(" ".join([line.decode("utf-8").strip() for line in file.readlines()]))
    return title, body

def to_line(data):
    title = data["title"]
    body = data["body"]
    genre_id = data["genre_id"]

    assert len(title) > 0 and len(body) > 0
    line = spacial_tokens_map["bos_token"] + title + \
           spacial_tokens_map["sep_token"] + body + \
           spacial_tokens_map["sep_token"] + f"{genre_id}" + spacial_tokens_map["eos_token"]
    return line

def createDataset():
    genre_files_list = [[] for genre in target_genres]
    all_data = []

    with tarfile.open(livedoor_dir) as archive_file:
        for archive_item in archive_file:
            for i, genre in enumerate(target_genres):
                if genre in archive_item.name and archive_item.name.endswith(".txt"):
                    genre_files_list[i].append(archive_item.name)

        for i, genre_files in enumerate(genre_files_list):
            for name in genre_files:
                file = archive_file.extractfile(name)
                title, body = read_title_body(file)
                title = normalize_text(title)
                body = normalize_text(body)

                if len(title) > 0 and len(body) > 0:
                    all_data.append({
                        "title": title,
                        "body": body,
                        "genre_id": i
                    })

    random.seed(123)
    random.shuffle(all_data)

    data_size = len(all_data)
    train_ratio, dev_ratio, test_ratio = 0.8, 0.1, 0.1

    with open(f"./dataset/train.txt", "w", encoding="utf-8") as f_train, \
         open(f"./dataset/dev.txt", "w", encoding="utf-8") as f_dev, \
         open(f"./dataset/test.txt", "w", encoding="utf-8") as f_test:
        
        for i, data in tqdm(enumerate(all_data)):
            line = to_line(data)
            if i < train_ratio * data_size:
                f_train.write(line + '\n')
            elif i < (train_ratio + dev_ratio) * data_size:
                f_dev.write(line + '\n')
            else:
                f_test.write(line + '\n')


if __name__ == '__main__':
    createDataset()