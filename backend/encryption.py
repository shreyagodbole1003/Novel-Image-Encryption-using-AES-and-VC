##ENCRYPT CODE

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

#Part 1 
#Reading input Image and encoding it using base64
with open("test3.jpg", "rb") as img_file:
    BI = base64.b64encode(img_file.read())
print(BI)
BI = BI.decode("utf-8")

#Part 2
# My key
K = ""
f = open('mykey.txt','r')
for i in f:
    K += i
f.close()

SK = hashlib.sha3_512(K.encode()) 
print("The hexadecimal equivalent of SHA3_512 is : ") 
print(SK.hexdigest())
len(SK.hexdigest())

#Part 3
# AES 256 in OFB mode:
# from Crypto.Cipher import AES
# from Crypto.Random import new as Random
# from hashlib import sha3_512
# from base64 import b64encode,b64decode

class AESCipher:
    def __init__(self,data,key):
        self.block_size = 16
        self.data = data
        self.key = sha3_512(key.encode()).digest()[:32]
        self.pad = lambda s: s + (self.block_size - len(s) % self.block_size) * chr (self.block_size - len(s) % self.block_size)
        self.unpad = lambda s: s[:-ord(s[len(s) - 1:])]

    def encrypt(self):
        plain_text = self.pad(self.data)
        iv = Random().read(AES.block_size)
        cipher = AES.new(self.key,AES.MODE_OFB,iv)
        return b64encode(iv + cipher.encrypt(plain_text.encode())).decode()


#Encrypting image using base 64 encoded text and hashed key - SHA256
#AES-256
c = AESCipher(BI,SK.hexdigest()).encrypt()
print(c)

#Part 4
w = 255
h = len(SK.hexdigest())

# creating new Image C of size(w,h) 
# initializing as blank
C = np.zeros((h,w,1), dtype = 'uint8')  #(26,255,1)

# Filling pixels in C
# the i takes in range of 26 lettrs currently in h
# j is the ascii order no. of current letter
# k is in the range of 255
# so for all values that are smaller than the order of character (j) 
# it will save values as 0
for i in range(h):
    j = ord(SK.hexdigest()[i])
    for k in range(w):
        if k < j:
            C[i][k][0] = 0
        else:
            C[i][k][0] = 255

filename = 'ToBeSent for Decryption/C.png'
cv2.imwrite(filename, C)

#Part 7
#taking ref img and the using it to overlap our key share on it
#that is: P is overlapped on a then saved in d
a = cv2.imread("test1.jpg")
cA, (cH, cV, cD)=pywt.dwt2(a,'haar',mode='constant')

for m in range(h):
    for n in range(w):
        cV[m][n][0] = math.floor(C[m][n][0]/10)   # 36  cv=3 cd 6
        cD[m][n][0] = C[m][n][0] % 10

d = pywt.idwt2((cA, (cH, cV, cD)),'haar')
d = np.uint8(d)

cv2.imshow("image", d)
cv2.waitKey(0)
cv2.destroyAllWindows()

filename = 'ToBeSent for Decryption/d.png'
cv2.imwrite(filename, d)

#Part 5
# Dividing C into R and P
# initializing R and P of same size as C
R = np.ones((h,w,3), dtype = 'uint8')    #h,w,3
P = np.ones((h,w,3), dtype = 'uint8')    #h,w,3

# filling the pixels of R
# i is in the range of len(key) i.e. 26
# j is in the range  of 255
# every first value in matrix of 255x3 is changed based on the normal 
for i in range(h):                #h
    for j in range(w):            #w
        r = np.random.normal(0,1,1)
        R[i][j][0] = r
        R[i][j][1] = r-(r/2)

# filling the pixels of P

for i in range(h):               #h
    for j in range(w):           #w
        p = R[i][j][0] ^ C[i][j][0]
        P[i][j][0] = p

plt.imshow(R)
plt.imshow(P)

filename = 'ToBeSent for Decryption/R.png'
cv2.imwrite(filename, R)
filename = 'ToBeSent for Decryption/P.png'
cv2.imwrite(filename, P)



#Part 6
#linear regression
# for i in P:
#     for j in i:
#         print(j)

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
    print(n1,"\n",n2)


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


xdf

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

sum(ydf['1']),sum(xdf['1'])

from sklearn.linear_model import LinearRegression
LRmodel = LinearRegression()
LRmodel.fit(xdf,ydf)

# z is for prediction
zdf = pd.DataFrame(columns = ['1','2'])
a = []
b = []
for i in C:
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

sum(zdf['1'])

predict = LRmodel.predict([[sum(zdf['1']),sum(zdf['2'])]])
predict

x = round(predict[0][0])%26
y = round(predict[0][1])%26
x,y

txt = []
for each in c:
    ch = ord(each)
    txt.append(int(ch))

text = ""
for t in txt:
    text += chr(t) + " "

print(text)

f = open("ToBeSent for Decryption/cipher.txt",'w',encoding='utf-8')
f.write(text)
f.close()