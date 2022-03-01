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

