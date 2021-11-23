# Day_02_01_MultiLayers.py
import tensorflow.keras as keras


# 퀴즈
# mnist의 load_data 함수 반환값을 x_train, x_test, y_train, y_test로 구분하세요
def show_mnist():
    mnist = keras.datasets.mnist.load_data()
    print(type(mnist))          # <class 'tuple'>
    print(len(mnist))           # 2

    print(type(mnist[0]), type(mnist[1]))   # <class 'tuple'> <class 'tuple'>
    print(len(mnist[0]), len(mnist[1]))     # 2 2

    print(type(mnist[0][0]), type(mnist[0][1]))     # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    print(type(mnist[1][0]), type(mnist[1][1]))     # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

    print(mnist[0][0].shape, mnist[0][1].shape)     # (60000, 28, 28) (60000,)
    print(mnist[1][0].shape, mnist[1][1].shape)     # (10000, 28, 28) (10000,)


# 퀴즈
# mnist 데이터셋에 대해 동작하는 딥러닝 모델을 구축하세요 (목표는 80% 이상)
def mnist_softmax():
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    # print(x_train.shape, x_test.shape)        # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape)        # (60000,) (10000,)

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    # print(x_train.shape, x_test.shape)        # (60000, 784) (10000, 784)

    # RGB : 0 ~ 255
    x_train = x_train / 255  # minmax scale
    x_test = x_test / 255

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


def mnist_multi_layers():
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    # print(x_train.shape, x_test.shape)        # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape)        # (60000,) (10000,)

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    # print(x_train.shape, x_test.shape)        # (60000, 784) (10000, 784)

    # RGB : 0 ~ 255
    x_train = x_train / 255  # minmax scale
    x_test = x_test / 255

    model = keras.Sequential()
    model.add(keras.layers.InputLayer([784]))
    # (?, 256) = (?, 784) @ (784, 256)
    model.add(keras.layers.Dense(256, activation='relu'))   # wx + b
    # (?, 64) = (?, 256) @ (256, 64)
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    # model.compile(optimizer=keras.optimizers.Adam(0.01),
    #               loss=keras.losses.sparse_categorical_crossentropy,
    #               metrics='acc')
    #
    # model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)
    # print(model.evaluate(x_test, y_test, verbose=0))


# show_mnist()

# mnist_softmax()
mnist_multi_layers()






