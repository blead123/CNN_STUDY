# Day_05_03_ResNet.py
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


# 퀴즈
# 케라스에서 제공하는 레즈넷50 모델을 우리가 사용하는 형태로 변환하세요 (summary 함수로 결과 확인)
def block1(x, filters, kernel_size=3, stride=1):
    shortcut = layers.Conv2D(4 * filters, 1, strides=stride)(x)
    shortcut = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(shortcut)

    x = layers.Conv2D(filters, 1, strides=stride)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME')(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(4 * filters, 1)(x)
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)

    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


def stack1(x, filters, blocks, stride1=2):
    x = block1(x, filters, stride=stride1)
    for i in range(2, blocks + 1):
        x = block1(x, filters)
    return x


img_input = layers.Input(shape=[224, 224, 3])

x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
x = layers.Conv2D(64, 7, strides=2, use_bias=False)(x)

x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
x = layers.Activation('relu')(x)

x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
x = layers.MaxPooling2D(3, strides=2)(x)

# --------------------------------- #

#  50 : 3, 4,  6, 3
# 101 : 3, 4, 23, 3
# 152 : 3, 8, 36, 3
x = stack1(x, 64, 3, stride1=1)
x = stack1(x, 128, 8)
x = stack1(x, 256, 36)
x = stack1(x, 512, 3)

# --------------------------------- #

x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
x = layers.Activation('relu')(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1000, activation='softmax')(x)

model = keras.Model(img_input, x)
model.summary()
