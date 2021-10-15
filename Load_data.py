import numpy as np
from keras_preprocessing import image
from PIL import Image
from numpy import hstack
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

path = './data'


def Get_MNIST_USPS_COMIC():
    data = 0
    if data == 1:
        x = scio.loadmat(path + "/MNIST-USPS.mat")
        print(x)
        x1 = x['X1']
        x2 = x['X2']
        Y = x['Y']
        print(x1.shape)
        print(x2.shape)
        print(Y.shape)
        print(x1[0])
        print(x2[0])
        print(Y[0])
        x1 = x1.reshape((5000, 28, 28))
        x2 = x2.reshape((5000, 16, 16), order='A')
        print(Y)
        Y = Y[0].reshape(5000,)
        print(Y)
        xu_reshape = np.zeros([len(x2), 28, 28], dtype=float)
        for i in range(len(x2)):
            for x in range(16):
                for y in range(16):
                    xu_reshape[i][x + 6][y + 6] = x2[i][x][y]

        print(x1.shape)
        print(xu_reshape.shape)
        print(Y.shape)
        z = np.linspace(0, len(Y) - 1, len(Y), dtype=int)
        np.random.shuffle(z)
        # print(z)
        # print(y_label)
        x_data_m = x1
        x_data_u = xu_reshape
        y_label = Y
        x_shuffle_m = np.copy(x_data_m)
        x_shuffle_u = np.copy(x_data_u)
        y_shuffle = np.copy(y_label)
        for i in range(len(y_label)):
            x_shuffle_m[i] = x_data_m[z[i]]
            x_shuffle_u[i] = x_data_u[z[i]]
            y_shuffle[i] = y_label[z[i]]
        x_shuffle_m = x_shuffle_m.reshape([-1, 28, 28, 1])
        x_shuffle_u = x_shuffle_u.reshape([-1, 28, 28, 1])/255
        print(x_shuffle_m.shape)
        print(x_shuffle_u.shape)
        print(y_shuffle.shape)
        print(x_shuffle_m[0])
        print(x_shuffle_u[0])
        # print(y_shuffle[0])
        scio.savemat(path + '/2V_MNIST_USPS.mat', {'X1': x_shuffle_m, 'X2': x_shuffle_u, 'Y': y_shuffle})
    data = scio.loadmat(path + "/2V_MNIST_USPS.mat")
    x1 = data['X1']
    x2 = data['X2']
    Y = data['Y'][0]
    ge = np.random.randint(0, len(x1), 1, dtype=int)
    image1 = np.reshape(x1[ge], (28, 28))
    image2 = np.reshape(x2[ge], (28, 28))
    print(Y[ge][0])
    plt.figure('Mnist')
    plt.imshow(image1)
    plt.show()
    plt.figure('USPS')
    plt.imshow(image2)
    plt.show()
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)

    return [x1, x2], Y


def BDGP():
    data = scio.loadmat(path + "/2V_BDGP.mat")
    x1 = data['X1']
    x2 = data['X2']
    Y = data['Y'][0]
    print(x1.shape)
    print(x2.shape)
    print(Y.shape)
    return [x1, x2], Y


def load_data_conv(dataset):
    print("load:", dataset)
    if dataset == 'MNIST_USPS_COMIC':             # MNIST-USPS
        return Get_MNIST_USPS_COMIC()
    elif dataset == 'BDGP':                       # BDGP
        return BDGP()
    else:
        raise ValueError('Not defined for loading %s' % dataset)
