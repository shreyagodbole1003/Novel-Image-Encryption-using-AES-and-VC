import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import new as Random
from hashlib import sha3_512
from base64 import b64encode, b64decode
import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pywt
import math
from sklearn.linear_model import LinearRegression

f = open("ToBeSent for Decryption/cipher.txt",'r',encoding='utf-8')
cipher = f.read()
cipher

# reading P and R images
P = cv2.imread('ToBeSent for Decryption/P.png')
R = cv2.imread('ToBeSent for Decryption/R.png')
f = cv2.imread('ToBeSent for Decryption/d.png')

#intializing h and w
h = np.shape(P)[0] #rows
w = np.shape(P)[1] #coloumn

cA, (cH, cV, cD)=pywt.dwt2(f,'haar',mode='constant')

CK = np.ones((h,w,3), dtype='uint8')

for m in range(h):
    for n in range(w):
        CK[m][n][0] = cV[m][n][0]*10 + cD[m][n][0] 


# initialize the image CK
# CK = np.ones((h,w,3), dtype = 'uint8')

# for i in range(h):
#     for j in range(w):
#         ck = g[i][j][0] ^ R[i][j][0] 
#         CK[i][j][0] = ck

K1 = []
for i in range(len(CK)):
    K1.append(0)

for i in range(len(CK)):
    count = 0
    for j in range(len(CK[i])):
        if CK[i][j][0] == 0: #counting the number of pixels which have 0 as value
            count += 1
    K1[i] = chr(count)

K1 = "".join(K1) # list into string
K1
print(K1)

SK1 = hashlib.sha3_512(K1.encode()) 

print("The hexadecimal equivalent of SHA256 is : ") 
print(SK1.hexdigest())

# AES 256 in OFB mode:

class AESCipher:
    def __init__(self,data,key):
        self.block_size = 16
        self.data = data
        self.key = sha3_512(key.encode()).digest()[:32]
        self.pad = lambda s: s + (self.block_size - len(s) % self.block_size) * chr (self.block_size - len(s) % self.block_size)
        self.unpad = lambda s: s[:-ord(s[len(s) - 1:])]

    def decrypt(self):
        cipher_text = b64decode(self.data.encode())
        iv = cipher_text[:self.block_size]
        cipher = AES.new(self.key,AES.MODE_OFB,iv)
        return self.unpad(cipher.decrypt(cipher_text[self.block_size:])).decode()


xdf = pd.DataFrame(columns = ['1','2'])
a = []
b = []
for i in P:
    k = 0
    n1 = []
    n2 = []
    for j in i:
        if k%2==0:
            n1.append(np.sum(j))
        else:
            n2.append(np.sum(j))
        k += 1    
    a.append(sum(n1))
    b.append(sum(n2))
xdf['1'] = a
xdf['2'] = b

ydf = pd.DataFrame(columns = ['1','2'])
a = []
b = []
for i in R:
    k = 0
    n1 = []
    n2 = []
    for j in i:
        if k%2==0:
            n1.append(np.sum(j))
        else:
            n2.append(np.sum(j))
        k += 1    
    a.append(sum(n1))
    b.append(sum(n2))
ydf['1'] = a
ydf['2'] = b

LRmodel = LinearRegression()
LRmodel.fit(xdf,ydf)

zdf = pd.DataFrame(columns = ['1','2'])
a = []
b = []
for i in CK:
    k = 0
    n1 = []
    n2 = []
    for j in i:
        if k%2==0:
            n1.append(np.sum(j))
        else:
            n2.append(np.sum(j))
        k += 1    
    a.append(sum(n1))
    b.append(sum(n2))
zdf['1'] = a
zdf['2'] = b

predict = LRmodel.predict([[sum(zdf['1']),sum(zdf['2'])]])

x = round(predict[0][0])%26
y = round(predict[0][1])%26

x,y

cipher = cipher.split(' ')
"".join(cipher)

txt = []
for each in cipher:
    try:
        ch = ord(each) #ord means we are converting back to ascii format
        txt.append(int(ch))
    except:
        print(each)

text = ""
for t in txt:
    text += chr(t) # converting ascii values into string

print(text)

de = AESCipher(text,K1).decrypt()
de = de.encode("utf-8")

with open("DecryptedImg.png", "wb") as fh:
    fh.write(base64.decodebytes(de))









