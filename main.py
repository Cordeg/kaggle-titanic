#!/usr/bin/env python
# coding: utf-8

# In[19]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import csv
from sklearn.model_selection import train_test_split

#inputdir = "/kaggle/input/titanic"
inputdir = "./input"

train_dirname = os.path.join(inputdir, 'train.csv')
test_dirname = os.path.join(inputdir, 'test.csv')

class InputData(object):
    
    def __init__(self, dirname):
        self.variables, self.data = self.load_data(dirname)
                
    def load_data(self, dirname):
        with open(dirname, 'r') as f:
            reader = csv.reader(f)
            contents = list(reader)
        
        variables = contents[0]
        data = np.array(contents[1:])
        
        return variables, data
    
    
class MyInputData(InputData):
    
    def __init__(self, dirname):
        super().__init__(dirname)
        
        # variables to use my model.
        # omit ['PassengerId', 'Name', 'Ticket', 'Cabin']
        self.used_variables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        
    def get_row(self, var):
        transposed_data = self.data.transpose()
        data_row = transposed_data[self.variables.index(var)]

        if var == 'Survived':
            data_row = [int(_) for _ in data_row]
        elif var == 'Sex':
            d = {'female':0, 'male':1, '': 2}
            data_row = [d[_] for _ in data_row]
        elif var == 'Embarked':
            d = {'C':0, 'Q':1, 'S': 2, '': 3}
            data_row = [d[_] for _ in data_row]
        elif var in ['Pclass', 'SibSp', 'Parch', 'PassengerId']:
            data_row = [int(_) for _ in data_row]
        else:
            # 今後の改善点：ゴミデータを含む場合は除去したした方が良い。
            data_row = [float(_) if _!='' else 0 for _ in data_row]        

        return np.array(data_row)
    
    def get_labels(self):
        var = 'Survived'
        if var in self.variables:
            return self.get_row(var)
        else:
            raise Exception('No labels exists.')
    
    def get_array(self):
        data_array = []
        for variable in self.used_variables:
            data_array.append(self.get_row(variable))

        return np.array(data_array).transpose()
    
    def train_test_split(self, shuffle=True):
        labels = self.get_labels()
        # list to transposed 1d array: [x, y, z] -> [[x], [y], [z], ...]
        labels = labels.reshape([-1,1])
        
        data_array = self.get_array()
        
        labels_and_data = np.hstack([labels, data_array])
        train_labels_and_data, test_labels_and_data = train_test_split(labels_and_data, shuffle=shuffle)
        
        train_labels = train_labels_and_data[:, 0]
        train_data = train_labels_and_data[:, 1:]
        test_labels = test_labels_and_data[:, 0]
        test_data = test_labels_and_data[:, 1:]
        
        # cast from float to int
        train_labels = train_labels.astype(int)
        test_labels = test_labels.astype(int)
        
        return train_labels, train_data, test_labels, test_data
        

input_data = MyInputData(train_dirname)
labels = input_data.get_labels()
data_array = input_data.get_array()

# test
print("Labels:", labels[:10])
print(input_data.used_variables)
print(data_array[0])
print(data_array[1])
tral, trad, tesl, tesd = input_data.train_test_split()
print(tral.shape, trad.shape, tesl.shape, tesd.shape)
print(tral[:10])


# In[67]:


from sklearn import tree

best_score = 0.0
best_classifier = None

for depth in range(3, 12):
    for _ in range(10):
        train_labels, train_data, test_labels, test_data = input_data.train_test_split()

        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(train_data, train_labels)
        
        predicted = clf.predict(test_data)
        score = sum(predicted == test_labels) / len(test_labels)
        if score > best_score:
            best_score = score
            best_classifier = clf
            print("depth:", depth)
            print("score:", score)


# In[59]:


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
pass


# In[70]:


input_data = MyInputData(test_dirname)
data_array = input_data.get_array()

ids = input_data.get_row('PassengerId')
ids = [int(_) for _ in ids]

print(ids[:10])

predicted = clf.predict(data_array)

out = list(zip(ids, predicted))
print(out[:10])


# In[69]:


def write_result(dirname):
    with open(dirname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["PassengerId", "Survived"])
        writer.writerows(out)
        
out_dirname = './out.txt'
write_result(out_dirname)


# In[ ]:




