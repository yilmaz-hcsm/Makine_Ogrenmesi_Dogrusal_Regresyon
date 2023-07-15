#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#1. gerekli kutuphaneleri dahil ediyoruz.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#veriyi işliyoruz
veriler = pd.read_csv('satislar.csv')



#veri on isleme
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]


#verileri train ve test olarak 66 ve 33 olarak ayırıyoruz
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)




# model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)# x_testten y_testi tahmin etti yani y_teste yakın değerleri tahmin etmiş oldu


x_train = x_train.sort_index()#sıralamazsak veriden veriye atlar tablo okunamaz hale gelir
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")







