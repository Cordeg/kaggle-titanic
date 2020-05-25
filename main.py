#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import csv

train_dirname = './input/train.csv'
test_dirname = './input/test.csv'

with open(train_dirname, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

train_labels = data[0]
train_data = np.array(data[1:])

# test
print(list(train_labels))
print(train_data[0])
print('len:', len(train_data[0]))
print(train_data[1])
print('len:', len(train_data[1]))

# %% [code]
# global varianbles: usel in "get_row" and "get_array"
base_labels = train_labels
base_data = train_data

# omit ['PassengerId', 'Name', 'Ticket', 'Cabin']
labels_of_model = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def get_row(label):
    transposed_data = base_data.transpose()
    data_row = transposed_data[base_labels.index(label)]
    
    if label == 'Survived':
        data_row = [int(_) for _ in data_row]
    elif label == 'Sex':
        d = {'female':0, 'male':1, '': 2}
        data_row = [d[_] for _ in data_row]
    elif label == 'Embarked':
        d = {'C':0, 'Q':1, 'S': 2, '': 3}
        data_row = [d[_] for _ in data_row]
    elif label in ['Pclass', 'SibSp', 'Parch']:
        data_row = [int(_) for _ in data_row]
    else:
        # 今後の改善点：ゴミデータを含む場合は除去したした方が良い。
        data_row = [float(_) if _!='' else 0 for _ in data_row]        

    return np.array(data_row)

def get_array(labels_of_model):
    data_array = []
    for label in labels_of_model:
        data_array.append(get_row(label))

    return np.array(data_array).transpose()

is_survived = get_row('Survived')
data_array = get_array(labels_of_model)

# test
print("Survived:", is_survived[:10])
print(labels_of_model)
print(data_array[0])
print(data_array[1])

# %% [code]
"""
Pclass: [1, 2, 3]
Sex: [0, 1, 2]
Age: [0~19, 20~39, 40~59, 60~]
SibSp: [0, 1, 2, 3, 4, 5~]
...0~8
Parch: [0, 1, 2, 3, 4~]
...0~6
'Fare': [0~50, 50~100, 100~150, 150~200, 200~]
...0~512
...154までに869人/891人
"""

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(data_array, is_survived)
predicted = clf.predict(data_array)

# %% [code]
sum(predicted == is_survived) / len(is_survived)

# %% [code]
with open(test_dirname, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

# rewrite global varianbles
base_data = np.array(data[1:])
base_labels = data[0]

ids = get_row('PassengerId')
ids = [int(_) for _ in ids]

data_array = get_array(labels_of_model)
predicted = clf.predict(data_array)

out = list(zip(ids, predicted))

out_dirname = './out.txt'

with open(out_dirname, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    writer.writerows(out)

