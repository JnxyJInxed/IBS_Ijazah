# -*- coding: utf-8 -*-
"""Ijazah_Basic_IBS_No_Library_WithAPI_02

"""### Data Operation"""

# Convert Data to Qr hidden data
def generateHiddenData(ID, S1_Compress, S2, date, copyNum):
  # date = '100423'
  # copyNum = '0000'
  s1Hex = '{:x}'.format(int(S1_Compress))
  s2Hex = '{:x}'.format(int(S2))
  hiddenQRData = str(ID) + s1Hex + '|' + s2Hex + str(date) + str(copyNum)
  return hiddenQRData

# Split variable from extracted QR data
def splitHiddenData(hiddenQRData):
  ID_extracted =  hiddenQRData[:5]

  Sign = hiddenQRData[5:-10]
  signComp = Sign.split('|')

  S1_Compress_Extracted = signComp[0]
  S1_Compress_Extracted = str(int(S1_Compress_Extracted, 16))
  S2_Extracted = signComp[1]
  S2_Extracted = int(S2_Extracted, 16)

  date = hiddenQRData[-10:-4]
  copyNum = hiddenQRData[-4:]

  # print(ID)
  # print(S1_Compress_Extracted == S1_Compress)
  # print(S2_Extracted == S2)
  # print(date)
  # print(copyNum)

  return ID_extracted, S1_Compress_Extracted, S2_Extracted, date, copyNum

"""### IBS Basic Operation"""

import hashlib
import math

#Cek syarat anti singular
def check_anti_singular(a,b):
  anti_singular = False
  if (4*(a**3) + 27*(b**2)) != 0 :
    anti_singular = True
  else:
    anti_singular = False
  return anti_singular

def multiInv_ECC(a,b):
  (xt,ct,yt) = (0,0,0)
  x1=1
  x2=0
  y1=0
  y2=1
  c1=x1*a+y1*b
  c2=x2*a+y2*b
  while c2>1 :
    var_pengali = c1//c2
    ct,xt,yt = c1,x1,y1
    c1,x1,y1 = c2,x2,y2

    c2 = ct-(c2*var_pengali)
    x2 = xt-(x2*var_pengali)
    y2 = yt-(y2*var_pengali)
  return y2

#Hash
def Hash(m):
  m_clean = m.replace(" ", "")
  m_clean = m.lower()
  print('m_clean :', m_clean)
  h = hashlib.sha3_256()
  h.update(str(m_clean).encode('utf-8'))
  h_int = int(h.hexdigest(), 16)
  print(h_int)
  return h_int


#Addition Point
#Addtion different point
def addition_ecc_different_point(poin1, poin2, p):
  xpoin1 = poin1[0]
  ypoin1 = poin1[1]
  xpoin2 = poin2[0]
  ypoin2 = poin2[1]

  ypoin = ypoin1 - ypoin2
  # print("ypoin : ",ypoin)
  xpoin = xpoin1 - xpoin2
  # print("xpoin : ",xpoin)
  if xpoin == 0:
    return (None, None)
  xinverse = multiInv_ECC(p, (xpoin % p))
  # print("xinverse : ",xinverse)
  # m = int((ypoin1 - ypoin2) * multiInv_ECC(p,(xpoin1 - xpoin2))) % p
  m = (ypoin * xinverse) % p
  # print("m : ",m)

  xr = ((m * m) - xpoin1 - xpoin2) % p
  # print("xr : ",xr)
  yr = (m * (xpoin1 - xr) - ypoin1) % p
  # print("yr : ",yr)

  return xr,yr

#Addition same point
def addition_ecc_same_point(poin1, p,a):
  xpoin1 = poin1[0]
  ypoin1 = poin1[1]
  # xpoin2 = poin2[0]
  # ypoin2 = poin2[1]

  atas = (3 * (xpoin1 * xpoin1) + a)
  # print("atas : ",atas)
  bawah = 2 * ypoin1
  # print("bawah : ",bawah)
  if bawah == 0:
    return (None, None)
  inversebawah = multiInv_ECC(p, (bawah % p))
  # print("inversebawah : ",inversebawah)
  # m = int((ypoin1 - ypoin2) * multiInv_ECC(p,(xpoin1 - xpoin2))) % p
  m = (atas * inversebawah) % p
  # print("m : ",m)

  xr = ((m * m) - (2 * xpoin1)) % p
  # print("xr : ",xr)
  yr = (m * (xpoin1 - xr) - ypoin1) % p
  # print("yr : ",yr)

  return xr,yr


#Subtraction Point
def subtraction_point(poin1, poin2, p, a):
  xpoin1 = poin1[0]
  ypoin1 = poin1[1]
  xpoin2 = poin2[0]
  ypoin2 = p - poin2[1]
  print('ypoin2', ypoin2)

  if (xpoin1 == xpoin2 and ypoin1 == ypoin2):
    print('SAME')
    return addition_ecc_same_point(poin1,p,a)
  else:
    return addition_ecc_different_point(poin1, (xpoin2, ypoin2), p)


#Join addition point between same and different point
def addition_point(poin1,poin2,a,p):
  xpoin1 = poin1[0]
  ypoin1 = poin1[1]
  xpoin2 = poin2[0]
  ypoin2 = poin2[1]

  if xpoin1 == xpoin2 and ypoin1 == ypoin2:
    xr,yr = addition_ecc_same_point(poin1, p,a)
  else:
    xr,yr = addition_ecc_different_point(poin1,poin2,p)

  return (xr,yr)

# multiplication point with optimization 02 with _doubling_adding
def muliplication_point(n, poin, p, a):
  n_binary = "{0:b}".format(n)

  P = poin
  for i in range(1, len(n_binary)):
    # print(n_binary[i])
    if (n_binary[i] == '1'):
      P = addition_ecc_same_point(P, p,a)
      P = addition_point(P, poin, a,p)
      # print('double-add, 1, P:', P)
    else:
      P = addition_ecc_same_point(P, p,a)
  return P

def _find_power_divisor(base, x, modulo=None):
  k = 0
  m = base
  while x % m == 0:
    k += 1
    m = pow(m * base, 1, modulo)
  return k

def legendre_symbol(a, p):
  """
  Calculate Legendre Symbol using Euler's criterion
  """
  if gcd(a, p) != 1:
    return 0
  d = pow(a, ((p - 1) // 2), p)
  if d == p - 1:
    return -1
  return 1

def _find_power(power_base, x, crib, modulo=None):
  k = 1
  r = power_base
  while pow(x, r, modulo) != crib:
    k += 1
    r *= power_base
  return k

def modinv(a, m):
  """
  Calculate Modular Inverse.
  - Find x satisfy ax \equiv 1 \mod m
  Args:
    a: target number
    n: modulus
  """
  if gcd(a, m) != 1:
    return 0
  if a < 0:
    a %= m
  return egcd(a, m)[1] % m

def gcd(x, y):
  """
  Calculate greatest common divisor
  """
  while y != 0:
      t = x % y
      x, y = y, t
  return x

def egcd(a, b):
  """
  Calculate Extended-gcd
  """
  x, y, u, v = 0, 1, 1, 0
  while a != 0:
    q, r = b // a, b % a
    m, n = x - u * q, y - v * q
    b, a, x, y, u, v = a, r, u, v, m, n
  return (b, x, y)

import random
def calculate_square_root_modular(a, m):
  # if is_prime(m):
  if legendre_symbol(a, m) == -1:
    return []
    # Tonelli-Shanks Algorithm
  if m % 4 == 3:
    r = pow(a, (m + 1) // 4, m)
    return [r, m - r]
  s = _find_power_divisor(2, m - 1)
  q = (m - 1) // 2**s
  z = 0
  while legendre_symbol(z, m) != -1:
    z = random.randint(1, m)
  c = pow(z, q, m)
  r = pow(a, (q + 1) // 2, m)
  t = pow(a, q, m)
  l = s
  while True:
    if t % m == 1:
      # assert (r ** 2) % m == a
      return [r, m - r]
    i = _find_power(2, t, 1, m)
    # print('i :', i)
    power = l - i - 1
    if power < 0:
      power = modinv(2**-power, m)
    else:
      power = 2**power
    b = pow(c, power, m)
    r = (r * b) % m
    t = (t * (b**2)) % m
    c = pow(b, 2, m)
    l = i
  if m == 2:
    return a
  if m % 4 == 3:
    r = pow(a, (m + 1) // 4, m)
    return [r, m - r]
  if m % 8 == 5:
    v = pow(2 * a, (m - 5) // 8, m)
    i = pow(2 * a * v, 2, m)
    r = a * v * (i - 1) % m
    return [r, m - r]
  if m % 8 == 1:
    e = _find_power_divisor(2, m - 1)
    q = (m - 1) // 2**e
    z = 1
    while pow(z, 2**(e - 1), m) == 1:
      x = random.randint(1, m)
      z = pow(x, q, m)
    y = z
    r = e
    x = pow(a, (q - 1) // 2, m)
    v = a * x % m
    w = v * x % m
    while True:
      if w == 1:
        return [v, m - v]
      k = _find_power(2, w, 1, m)
      d = pow(y, 2**(r - k - 1), m)
      y = pow(d, 2, m)
      r = k
      v = d * v % m
      w = w * y % m

"""### Curve Parameter"""

from collections import namedtuple
# Preparing the curve
ECData = namedtuple('ECData', ['p', 'a', 'b', 'n', 'Gx', 'Gy', 'h'])
# Let a eliptic curve defined by y^2 = x^3 +ax+b defined over Fp : E[Fp]
# genertade by basepoint <P> with an order of #E[Fp]
    # p = EC defined over Fp
    # a = constant for y^2 = x^3 +ax+b
    # b= constant for y^2 = x^3 +ax+b
    # Gx = x value of base point
    # Gy = y value of base point
    # n = order of [h]E[Fp] = <G> = [h]<P>
    # h = G is a element h-torsion of E[Fp]

bn158 = ECData(
    p = 0x24240D8241D5445106C8442084001384E0000013,
    a= 0x0000000000000000000000000000000000000000,
    b= 0x0000000000000000000000000000000000000011,
    Gx = 0x24240D8241D5445106C8442084001384E0000012,
    Gy = 0x0000000000000000000000000000000000000004,
    n = 0x24240D8241D5445106C7E3F07E0010842000000D,
    h = 0x01
)

toyF13 = ECData(
    p = 13,
    a= 1,
    b= 0,
    Gx = 6,
    Gy = 12,
    n = 10,
    h = 20
)

typeG = ECData(
    p = 205523667896953300194896352429254920972540065223,
    # p = 205523667896953300194896352429254920972540065223,

    # a= 465197998498440909244782433627180757481058321,
    a = 0x14DC360C51B72FC7CB5C8749F98592DC827011,

    b= 463074517126110479409374670871346701448503064,
    # b= 463074517126110479409374670871346701448503064

    Gx = 12383835871835950148756501359909159276513424558,
    Gy = 190063739341072368115073423604198992137643129204,
    n = 503189899097385532598571084778608176410973351,
    h = 0x01
)

p224 = ECData(
    p = 26959946667150639794667015087019630673557916260026308143510066298881,
    a= -3,
    b= 18958286285566608000408668544493926415504680968679321075787234672564,
    Gx = 19277929113566293071110308034699488026831934219452440156649784352033,
    Gy = 19926808758034470970197974370888749184205991990603949537637343198772,
    n = 26959946667150639794667015087019625940457807714424391721682722368061,
    h = 1
)

sect163k1 = ECData(
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFEE37,
    a= 0,
    b= 3,
    Gx = 0xDB4FF10EC057E9AE26B07D0280B7F4341DA5D1B1EAE06C7D,
    Gy = 0x9B2F2F6D9C5628A7844163D015BE86344082AA88D95E2F9D,
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFE26F2FC170F69466A74DEFD8D,
    h = 1
)

data = sect163k1
a = data.a
b = data.b
base_poin = (data.Gx, data.Gy)
n = data.n
p = data.p
P = base_poin

"""### IBS Scheme

#### Helper Function
"""

# GET CORRESPONDING Y
def get_corresponding_y(x):
  y_pow2 = (pow(x,3) + (a*x) + b)
  print("y_pow2 = ", y_pow2)
  # y_2 = n % p
  # print('x,y_2', x,',',y_2)
  # akarkan y^2nya
  y1,y2 = calculate_square_root_modular(y_pow2, p)
  print("x = ", x, ",", "y1 = ", y1)
  print("x = ", x, ",", "y2 = ", y2)
  print("")
  return y1, y2

def pointCompression(P):
  # get point x coordinate from the point
  # return
  if (P[1]%2 == 0): # y is even
    compressed = str(0) + str(int(P[0]))
  else:
    compressed = str(1) + str(int(P[0]))
  return compressed

def pointDecompression(Px):
  # return point of ECC from the x coordinate
  print("Px", Px)
  leastBit = int(Px[0])
  print('leastBit :', leastBit)

  xVal = int(Px[1:])
  print('xVal :', xVal)

  Py = get_corresponding_y(xVal)
  print(Py)

  if (Py[0]%2 == leastBit):
    yVal = Py[0]
  else:
    yVal = Py[1]
  print('yVal :', yVal)

  P = (xVal, yVal)
  return P

def getRmid():
  from random import SystemRandom
  cryptogen = SystemRandom()
  randInt = cryptogen.randrange(int((n-1)/2))
  return randInt

def H1(m):
  # H1 = a futnion to map the message or binary string to the Field (Zp) element
  # h = Hash(m)%n
  print('\tH1')

  h = int(Hash(str(m)))
  print('\t\th :', h)

  h_int = h%n
  print('\t\th_int :', h_int)
  print()
  return h_int

def H2(m1, m2, G1, G2):
  # Hash function H2 will map the space of all TWO bit strings ({0,1}*) and TWO element in G (G) into {1,..l-1}
  print('\tH2')

  h1 = int(Hash(m1))%n
  print('\t\th1 :', h1)

  h2 = int(Hash(m2))%n
  print('\t\th2 :', h2)

  G1_ = muliplication_point(h1, G1, p, a)
  print('\t\tG1_ :', G1_)

  G2_ = muliplication_point(h2, G2, p, a)
  print('\t\tG2_ :', G2_)

  xVal = G1_[0] + G2_[0]
  print('\t\txVal :', xVal)
  print()

  xVal_int = int(xVal)%n
  return xVal_int

"""#### Main Function"""

def setup(s):
  print('SETUP')
  # setup by KGC
  # s is master secret paramater; s = {1,..q-1}
  # Pub is master public paraemetr; Pub = s*generator point in choosen curve
  Pub = muliplication_point(s, P, p, a) #s*P
  return Pub

def extract(ID, s):
  # – CA computes
  # public key: QID = H1(ID) ∈ Zq∗.
  # – CA computes
  # private key: dID = (rID + sQID) ∈ Zq∗.
  # – CA sends dID to the signer securely.
  # – CA also compute V_ID = rID*P  and Z_ID = d_ID*P ∈ G.
  # – CA announces ZID and
  # VID as public parameter.

  Q_id =  H1(str(ID))

  # rID = getRmid()%p
  rID = 7 #buat testing dengan bang hanang
  d_ID = ((rID+s)*Q_id)%p

  V_ID = muliplication_point(rID, P, p, a) #rID*P

  Z_ID = muliplication_point(d_ID, P, p, a) #d_ID*P

  return d_ID, V_ID, Z_ID

def sign(ID, m, d_ID, Z_ID, Pub):
  # User sign the message using geenrtae dsecret key VA and they ID
  # Pick a random number r_mid E Z/L* //r E {0,1,..l-1}
  # at random corresponding to m and ID

  x_ID = getRmid()%n
  print('x_ID = ', x_ID)

  S1 = muliplication_point(x_ID, Z_ID, p, a) #x_ID*Z_ID
  print("S1", S1)

  h = H2(m, ID, Z_ID, Pub)

  S2 = ((x_ID + h)%p)*d_ID

  S1_Compress = pointCompression(S1)

  return S1_Compress, S2

def verify(S1, S2, m, ID, Z_ID):
  print('VERIFY')
  # Receiver verify the user auth and the message interity

  S1 = pointDecompression(S1)
  print("S1", S1)
  # Z_ID = pointDecompression(Z_ID)

  h = H2(m, ID, Z_ID, Pub)

  print('a1')
  a1 = muliplication_point(S2, P, p, a)

  print('temp')
  temp = muliplication_point(h, Z_ID, p, a)

  print('a2')
  a2 =  addition_point(S1,temp,a,p)

  verifyStat = (a2 == a1)
  return verifyStat

"""### ON START UP"""

# Process below are the result from done on Trusted KeY Gneration Center
Pub = (19399464229459456007477471411003978864755290924325272939384776426428, 26863117366256785198219971769961599705632546399319105640204669897254)

# data for "ADW00"
d_ID = 939797642745129237523087870057529014097764531888855370378
V_ID = (1389498872671557601749818509830484747372342977419298411889,
         4132492685732390198376426404949785382972775552647596672764)
Z_ID = (2664358637641010344307070931379391014582722336309543772284, 926914216101337219991992490279981614974331699751485954931)

"""### FAST API

#### Library
"""

from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
import pickle
import json
import base64
from collections import defaultdict
import time
import asyncio

from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.templating import Jinja2Templates

from pprint import pprint

"""#### Controller"""

# Receive combined visible message and ID of signer
def CreateQRHiddenData(recieved_message, recieved_ID):

  S1_Compress_calculated, S2_calculated = sign(recieved_ID, recieved_message, d_ID, Z_ID, Pub)

  import datetime
  date = str(datetime.datetime.today().strftime('%d%m%y'))

  hiddenQRData = generateHiddenData(recieved_ID, S1_Compress_calculated, S2_calculated, date, '0000')

  return hiddenQRData

def verifyQRHiddenData(receivedQRData):
  strList = receivedQRData.split('##')
  recieved_message = strList[0]
  received_hiddenmessage = strList[1]
  # print('HEREEEEEEEEEEEEEEEEEEE')

  ID_extracted, S1_Compress_Extracted, S2_Extracted, date, copyNum = splitHiddenData(received_hiddenmessage)

  verifyStat = verify(S1_Compress_Extracted, S2_Extracted, recieved_message, ID_extracted, Z_ID)
  return verifyStat

"""#### Main FAST API"""

app = FastAPI()

"""#### Model"""

class dataGenerateQRInput(BaseModel):
  message: str
  ID_signer: str

class dataGenerateQROutput(BaseModel):
  hiddenQRmessage: str

class dataVerifyQRInput(BaseModel):
  messageQR: str

class dataVerifyQROutput(BaseModel):
  verifyStat: str

"""#### Route"""

@app.get('/')
def index():
  from datetime import datetime

  time = datetime.now()
  message = 'This is the homepage of the API at ' + str(time)

  return {'message': message}

@app.post("/generateQRData")
async def generateQRData(data :dataGenerateQRInput):
  pprint(data)
  message = data.message
  userID = data.ID_signer

  hiddenQRData = CreateQRHiddenData(message, userID)
  print('hiddenQRData : ', hiddenQRData)

  dataReturn = dataGenerateQROutput(
      hiddenQRmessage = str(hiddenQRData)
  )

  return dataReturn

@app.post("/verifyQRData")
async def verifyQRData(data :dataVerifyQRInput):
  pprint(data)
  dataQR = data.messageQR
  # print('HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE 01')
  # print('message', dataQR)

  verifyStat = verifyQRHiddenData(dataQR)
  print('verifyStat : ', verifyStat)

  dataReturn = dataVerifyQROutput(
      verifyStat = str(verifyStat)
  )

  return dataReturn
