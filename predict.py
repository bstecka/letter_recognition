# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
import warnings
from matplotlib import pyplot as plt

def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    pass

def load_data():
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pkl.load(f)

warnings.filterwarnings('ignore')
PICKLE_FILE_PATH = 'train.pkl'
data = load_data()
x_train = data[0] # N x D {0, 1} 30134 x 3136
y_train = data[1] # 1000 {1, 36} (23 letters + 10 digits + ?)
nw = 30000 # 13: R, 34: D, 2: O, 0: 6 - seemingly random
x0 = x_train[nw] # one row = one picture. 56 * 56 = D = 3136
print(y_train[nw])
x0 = np.reshape(x0,(56,56))
plt.imshow(x0)
plt.draw()
plt.waitforbuttonpress(0)