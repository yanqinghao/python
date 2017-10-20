# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:13:46 2017

@author: yqh
"""

#分开添加索引

import numpy as np
b = np.array([[1, 1], [2, 1], [3, 1], [4, 10], [5, 1],
              [6, 4], [7, 4], [8, 9], [9, 9], [10, 9]])
a = np.array([])
a = b[:,1]
c = np.array([])
c = b[:,0]
flag = []

i = 1
n = 0
temp =a[0]
while i <= (len(a)-1):
     if temp == a[i]:
        n +=1
     else:
        temp = a [i]
        n = 0
     if n >2:
         flag.append(i)
     else:
         if  n==2:
             flag.extend(list(range(i-2,i+1)))
     i += 1

print(np.delete(b,flag,axis=0))

#一起添加索引

import numpy as np
a = np.array([])

b = np.array([[1, 1], [2, 1], [3, 1], [4, 10], [5, 1],
              [6, 4], [7, 4], [8, 1], [9, 9], [10, 1]])

a = b[:,1]

l = len(a)
flag = []

i = j = 0
while i < l - 1:
    if a[i] == a[i + 1]:
        i += 1
    else:
        if i + 1 - j >= 3:
            flag.extend(list(range(j, i + 1)))
        i += 1
        j = i

# print(flag)
print(np.delete(b,flag,axis=0))
