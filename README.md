# deep learning 2022

## Overview
Fine tune T5 by livedoor news corpus

## Usage
**Prepare**
```
!wget -O ./model/ldcc-20140209.tar.gz https://www.rondhuit.com/download/ldcc-20140209.tar.gz
```
**Create dataset**
```
python createDataset.py
```
**Train**
```
python train.py
```
**Test**
```
python test.py
```

## Reference
[google/mt5-small](https://huggingface.co/google/mt5-small)

[livedoor news corpus](https://www.rondhuit.com/download.html#ldcc)

[t5-japanese-classification](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_classification.ipynb)
