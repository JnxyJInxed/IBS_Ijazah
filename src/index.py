# -*- coding: utf-8 -*-
"""ECC_PYTHON_OPERATION_AND_SCHEME06

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pC0By14KY6-IGNW_JL8Dc7l5Pv8sNZah

## install library (unecessary for server)
"""


"""## ECC_UTILS

### Basic operation
"""

import hashlib
import math  

#Cek syarat anti singular
def check_anti_singular(a,b):
  anti_singular = False
  if (4*(a**3) + 27*(b**2)) != 0 :
    print('Memenuhi syarat')
    anti_singular = True
  else:
    print('Tidak memenuhi syarat, masukan nilai lain')
    anti_singular = False
  return anti_singular


#List Point
# def list_point(check_singularity, a, b, p):
#   pointCollection = []
#   i = 0

#   if check_singularity:
#   # setelah memenuhi syarat, list point
#     for x in range(0, p):
#       i=i+1
#       n = (pow(x,3) + (a*x) + b)
#       print("n = ",n)
#       # y_2 = n % p
#       # print('x,y_2', x,',',y_2)
#       # akarkan y^2nya
#       y1,y2 = calculate_square_root_modular(n, p)
#       print("x = ", x, ",", "y1 = ", y1)
#       print("x = ", x, ",", "y2 = ", y2)
#       print("")
#       if (y1 != "doesnt exist"):
#         if y1 == y2:
#           pointCollection.append([x,y1])
#         else:
#           pointCollection.append([x,y1])
#           pointCollection.append([x,y2])
#   return pointCollection, i


#Multi Inverse Using Extended Euclidean
## mencari pasangan x dari a yang akan membuat
## a*x mod p == 1
#PAke extended euclidian

# x1*a+y1*b=c1
# x2*a+y2*b=c2  x var_pengali floor(c1/c2)
# --------------------  -
# x3*a+y3*b=c3

def multiInv_ECC(a,b):
  (xt,ct,yt) = (0,0,0)
  x1=1
  x2=0
  y1=0
  y2=1
  c1=x1*a+y1*b
  c2=x2*a+y2*b
  while c2>1 :
    # print(x1,'*',a,'+',y1,'*',b,'=',c1)
    var_pengali = c1//c2
    # print(x2,'*',a,'+',y2,'*',b,'=',c2,'    X', var_pengali)
    # print('------------------------ -')
    ct,xt,yt = c1,x1,y1
    c1,x1,y1 = c2,x2,y2

    c2 = ct-(c2*var_pengali)
    x2 = xt-(x2*var_pengali)
    y2 = yt-(y2*var_pengali)
  #   print(x2,'*',a,'+',y2,'*',b,'=',c2)
  #   print('')
  # print('       ',y2,'*',b,'=',c2,'+',-x2,'*',a)
  # print('       ','[',y2,']','[',b,']=[',a,']')
  # print('')
  # print('x*a+y*b=1; x = ', x2, '; y=',y2)
  # if y2<0:
  #   y2 = p-y2
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

# # Multiplication point
# def muliplication_point(n, poin, p, a):
#   result = poin
#   for i in range (1, n):
#       result = addition_point(result, poin, a, p)
#       # print("i ke : ",i, "hasilnya : ",result)
#   return result

# # optimization with _doubling_adding
# def muliplication_point(n, poin, p, a):
 
#   two_power_divisor = int(math.log(n, 2))
#   print('two_power_divisor :', two_power_divisor)

#   res = n-pow(2, two_power_divisor)
#   print('res :', res)

#   # P = 1
#   P = poin
#   for i in range(0, two_power_divisor):
#     # P = 2*P
#     P = addition_ecc_same_point(P, p,a)
    
#   print('P:', P)
  
#   # R = 1
#   R = poin
#   for i in range(0, res-1):
#     R = R + R
#     R = addition_point(R, poin, a,p)
#   print('R:', R)
  
#   Q = addition_point(P, R, a,p)
#   return Q

# optimization 02 with _doubling_adding
def muliplication_point(n, poin, p, a):
  n_binary = "{0:b}".format(n)
  # print('n_binary:', n_binary)

  # P = 1
  # for i in range(1, len(n_binary)):
  #   print(n_binary[i])
  #   if (n_binary[i] == '1'):
  #     P = 2*P
  #     P = P+1
  #     print('double-add, 1, P:', P)
  #   else:
  #     P = 2*P
  #     print('double, 0, P:', P)
  # print(P)
  #   # print()

  P = poin
  for i in range(1, len(n_binary)):
    # print(n_binary[i])
    if (n_binary[i] == '1'):
      P = addition_ecc_same_point(P, p,a)
      P = addition_point(P, poin, a,p)
      # print('double-add, 1, P:', P)
    else:
      P = addition_ecc_same_point(P, p,a)
  #     print('double, 0, P:', P)
  # print(P)
  return P

#Subgroup
# def subgroup(a, b, p, poin):
#   i = 1
#   result = poin
#   arr_result = []

#   while i < p and result != (None, None):
#     print("i = :",i)
#     result = muliplication_point(i,poin, p, a)
#     arr_result.append(result)
#     i += 1
    
#     print()
#     print('result', result)
#     print('----------------------')
#   return arr_result

# #Square Root Modulo
# #list reference
# #https://rosettacode.org/wiki/Tonelli-Shanks_algorithm
# #https://www.rieselprime.de/ziki/Modular_square_root
# def _find_power_divisor(base, x, modulo=None):
#   k = 0
#   m = base
#   while x % m == 0:
#     print('k :', k)
#     print(x,'%',2,'^',k,' =', x % m)
#     print()
#     k += 1
#     m = pow(m * base, 1, modulo)
  
#   print('k :', k)
#   print(x,'%',2,'^',k,' =', x % m)

#   return k

# def calculate_square_root_modular(a, m):
#   #check a^((m-1)/2) % m == 0
#   if pow(a, (m - 1) // 2, m) == 0:
#     r1 = 0
#     r2 = 0
#     return r1, r2
#   #check a^((m-1)/2) % m == 1
#   if pow(a, (m - 1) // 2, m) == 1:
#     #Modulus equal to 2
#     if m % 2 == 0:
#       r1 = r2 = r #"??"
#       return r1,r2
#     #Modulus congruent to 3 modulo 4
#     if m % 4 == 3:
#       r1 = pow(a, int((m + 1) // 4), m)
#       r2 = pow(-a, int((m + 1) // 4), m)
#       return r1,r2
#     #Modulus congruent to 5 modulo 8
#     if m % 8 == 5:
#       v = pow((2 * a), int((m - 5) / 8), m)
#       print("v = ",v)
#       i = (2 * a * (v ** 2)) % m
#       print("i = ",i)
#       r1 = (a * v * (i - 1)) % m
#       print("r1 = ",r1)
#       r2 = -(a * v * (i - 1)) % m
#       print("r2 = ", r2)
#       return r1,r2
#     if m % 8 == 1:
#       e = _find_power_divisor(2, m - 1)
#       q = (m - 1) // 2 ** e
#       x = 2
#       while x > 1 and x < m :
#         z = (x ** q) % m
#         print("initial z :)",z)
#         if (z ** (2 ** (e-1))) % m == 1:
#           z2 = (z ** (2 ** (e-1))) % m
#           print("z^2^e-1 : ",z2)
#           x += 1
#         else:
#           z2 = (z ** (2 ** (e-1))) % m
#           print("z^2^e-1 : ",z2)
#           break

#       y = z
#       print("y : ",y)
#       r = e
#       print("r : ",r)
#       x = pow(a, int((q - 1) // 2 ), m)
#       print("x : ",x)
#       v = (a * x) % m
#       print("v : ",v)
#       w = (v * x) % m
#       print("w : ",w)

#       while w != 1:
#         k = 0
#         while (w ** (2 ** (k))) % m != 1:
#           k += 1

#         print("k : ",k)

#         d = (y ** (2 ** (r - k - 1))) % m
#         print("d : ", d)
#         y = pow(d, 2, m)
#         print("y : ", y)
#         r = k
#         print("r : ",r)
#         v = (d * v) % m
#         print("v : ",v)
#         w = (w * y) % m
#         print("w : ",w)


#       v1 = v % m
#       v2 = -(v) % m
#       print(v1,v2)
#       return v1,v2
#       # q = (m - 1) // 2**e
#       # z = 1
#       # while pow(z, 2**(e - 1), m) == 1:
#       #   x = random.randint(1, m)
#       #   z = pow(x, q, m)
#       # y = z
#       # r = e
#       # x = pow(a, (q - 1) // 2, m)
#       # v = a * x % m
#       # w = v * x % m
#       # while True:
#       #   if w == 1:
#       #     return [v, m - v]
#       #   k = _find_power(2, w, 1, m)
#       #   print('k :', k)
#       #   print('w :',w)
#       #   print('m :',m)
#       #   # while (w ** (2 ** (k))) % m != 1:
#       #   #   k += 1
#       #   d = pow(y, 2**(r - k - 1), m)
#       #   y = pow(d, 2, m)
#       #   r = k
#       #   v = d * v % m
#       #   w = w * y % m
#   else:
#     return "doesnt exist", "doesnt exist"

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

"""### Curve Parameter

"""

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

data = p224   
a = data.a
b = data.b
base_poin = (data.Gx, data.Gy)
n = data.n
p = data.p
P = base_poin

"""## ECDSA SCHEME

#### Scheme Procedure
"""

import random
#Verifying ECDSA
def Verifying_ECDSA(r, s, m, QA):
  # 1. Memverifikasi bahwa r dan s adalah bilangan bulat yang antara [1,n-1]
  # 2. Menghitung h = Hash (m)
  # print('r', r)
  # print('s', s)

  print("verifying ECDSA")
  QA = pointDecompression(QA)
  print('QA', QA)

  h = Hash(m)%n
  print('h', h)

  inv_s = multiInv_ECC(n,s)%n
  c = inv_s
  print('c', c)

  u1=(h*c)%n
  print('u1',u1)

  u2=(r*c)%n
  print('u2',u2)

  try:
    u1p = muliplication_point(u1,base_poin, p, a)
    print("u1p = ",u1p)

    u2qa = muliplication_point(u2,QA, p, a)
    print("u2qa = ",u2qa)
  
  
    # O = u1*P + u2*QA
    O = addition_point(u1p,u2qa,a,p)
    print('O',O)
    
  except:
    O = (0,0)

  res = int(O[0])%n

  # 7. Menerima tanda tangan jika dan hanya jika v = r
  verifyStat = (res == r)
  print('Signature validity: ', verifyStat)
  return verifyStat

"""# Modified Scheme [Ref](https://ssrn.com/abstract=3351027)

## Scheme Process
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
  randInt = cryptogen.randrange(n-1)
  return randInt

"""### Verify"""

"""
– Verifier computes h = H2(m||ZID) ∈ Zq∗
– Verifier accept the signature Sig(S1, S2) on message
m if following holds: S2*P = S1 + hZID
"""
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
  
def verify(S1, S2, m, ID, Z_ID):
  print('VERIFY')
  # Receiver verify the user auth and the message interity

  S1 = pointDecompression(S1)
  Z_ID = pointDecompression(Z_ID)

  h = H2(m, ID, Z_ID, Pub)
  print('h :', h)

  a1 = muliplication_point(S2, P, p, a)
  print('a1 :', a1)
  
  temp = muliplication_point(h, Z_ID, p, a)
  print('temp :', temp)

  a2 =  addition_point(S1,temp,a,p)
  print('a2 :', a2)

  verifyStat = (a2 == a1)
  return verifyStat

"""# ON "STARTUP"
"""

# Process below are the result from done on Trusted KeY Gneration Center
Pub = (19399464229459456007477471411003978864755290924325272939384776426428, 26863117366256785198219971769961599705632546399319105640204669897254)

"""# Fast API

## Library
"""

# import sys
# sys.path.insert(0,'/content/drive/MyDrive/Project/Proctoring_Unjani/Collab WorkSpace/v003')

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

# !ngrok authtoken 2N34fPA7UpJM351Rnf0RXcsptLg_2eiEhTckNc41akVPqizZi

"""## MAIN"""

app = FastAPI()
# # app.mount("/testUI", StaticFiles(directory="/content/drive/MyDrive/Project/QR IJAZAH/UI/"), name="UI")
# # app.mount("/static", StaticFiles(directory="/content/drive/MyDrive/Project/QR IJAZAH/static/"), name="static")
# app.mount(
#     "/static", StaticFiles(directory="/content/drive/MyDrive/Project/QR IJAZAH/static"), name="static")

# templates = Jinja2Templates(directory="/content/drive/MyDrive/Project/QR IJAZAH/UI")

"""### Model"""

class ImageInDetect(BaseModel):
  id: int
  data: str

class dataExtractKeyInput(BaseModel):
  userID: str

class dataExtractKeyOutput(BaseModel):
  d_ID   : str
  V_ID  : str
  Z_ID   : str

class dataSignInput(BaseModel):
  userID: str
  message: str
  d_ID   : str
  Z_ID   : str

class dataSignOutput(BaseModel):
  s1: str
  s2: str

class dataVerifyInput(BaseModel):
  userID: str
  message: str
  s1   : str
  s2   : str
  Z_ID   : str

class dataVerifyOutput(BaseModel):
  verifyStat: str

class dataVerifyECDSAInput(BaseModel):
  message: str
  r   : str
  s   : str
  QA   : str

class dataVerifyECDSAOutput(BaseModel):
  verifyStat: str

"""### Controller

### Route
"""

@app.get('/')
def index():
    from datetime import datetime
    
    time = datetime.now()
    message = 'This is the homepage of the API at ' + str(time)

    return {'message': message}

@app.post("/verify")
async def verifyMessage(data :dataVerifyInput):
    pprint(data)
    userID = data.userID
    message = data.message
    s1  = data.s1
    s2  = int(data.s2)

    Z_ID = data.Z_ID

    verifyStat = verify(s1, s2, message, userID, Z_ID)
    
    # try:
    #   verifyStat = verify(s1, s2, message, userID, Z_ID)
    # except:
    #   verifyStat = False

    dataReturn = dataVerifyOutput(
        verifyStat = str(verifyStat)
    )

    return dataReturn

@app.post("/verifyECDSA")
async def verifyMessage(data :dataVerifyECDSAInput):
    pprint(data)
    message = data.message
    r  = int(data.r)
    s  = int(data.s)

    QA = data.QA

    verifyStat = Verifying_ECDSA(r, s, message, QA)
    # try:
    #   verifyStat = Verifying_ECDSA(r, s, m, QA)
    # except:
    #   verifyStat = False

    dataReturn = dataVerifyECDSAOutput(
        verifyStat = str(verifyStat)
    )

    return dataReturn

"""### Run server"""