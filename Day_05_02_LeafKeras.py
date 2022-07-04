# Day_05_02_LeafKeras.py
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing


# 퀴즈
# 5-1번 파일에서 진행했던 프로젝트를 케라스 버전으로 다시 구축하세요
def get_train():
    leaf = pd.read_csv('data/leaf_train.csv', index_col=0)
    # print(leaf)

    x = leaf.values[:, 1:]
    # print(x.dtype)            # object

    x = np.float32(x)
    # print(x.shape)            # (990, 192)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(leaf.species)
    # print(y[:10])             # [ 3 49 65 94 84 40 54 78 53 89]

    return x, y, enc.classes_


def get_test():
    leaf = pd.read_csv('data/leaf_test.csv', index_col=0)
    # print(leaf)
    # print(leaf.values.dtype)      # float64

    # print(leaf.index)
    # print(leaf.index.values)      # [   4    7    9   12  ...]

    return np.float32(leaf.values), leaf.index.values


x_train, y_train, classes = get_train()
x_test, s_ids = get_test()

model = keras.Sequential()
model.add(keras.layers.Dense(len(classes), activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2,
          validation_split=0.2)

p = model.predict(x_test)
# print(p.shape)            # (594, 99)
# print(p[:3])              # [[0.00881641 0.00859881 0.0057183 ...] [...] [...]]

f = open('data/leaf_submission_keras.csv', 'w', encoding='utf-8')

# columns = 'id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,Acer_Opalus,Acer_Palmatum,Acer_Pictum,Acer_Platanoids,Acer_Rubrum,Acer_Rufinerve,Acer_Saccharinum,Alnus_Cordata,Alnus_Maximowiczii,Alnus_Rubra,Alnus_Sieboldiana,Alnus_Viridis,Arundinaria_Simonii,Betula_Austrosinensis,Betula_Pendula,Callicarpa_Bodinieri,Castanea_Sativa,Celtis_Koraiensis,Cercis_Siliquastrum,Cornus_Chinensis,Cornus_Controversa,Cornus_Macrophylla,Cotinus_Coggygria,Crataegus_Monogyna,Cytisus_Battandieri,Eucalyptus_Glaucescens,Eucalyptus_Neglecta,Eucalyptus_Urnigera,Fagus_Sylvatica,Ginkgo_Biloba,Ilex_Aquifolium,Ilex_Cornuta,Liquidambar_Styraciflua,Liriodendron_Tulipifera,Lithocarpus_Cleistocarpus,Lithocarpus_Edulis,Magnolia_Heptapeta,Magnolia_Salicifolia,Morus_Nigra,Olea_Europaea,Phildelphus,Populus_Adenopoda,Populus_Grandidentata,Populus_Nigra,Prunus_Avium,Prunus_X_Shmittii,Pterocarya_Stenoptera,Quercus_Afares,Quercus_Agrifolia,Quercus_Alnifolia,Quercus_Brantii,Quercus_Canariensis,Quercus_Castaneifolia,Quercus_Cerris,Quercus_Chrysolepis,Quercus_Coccifera,Quercus_Coccinea,Quercus_Crassifolia,Quercus_Crassipes,Quercus_Dolicholepis,Quercus_Ellipsoidalis,Quercus_Greggii,Quercus_Hartwissiana,Quercus_Ilex,Quercus_Imbricaria,Quercus_Infectoria_sub,Quercus_Kewensis,Quercus_Nigra,Quercus_Palustris,Quercus_Phellos,Quercus_Phillyraeoides,Quercus_Pontica,Quercus_Pubescens,Quercus_Pyrenaica,Quercus_Rhysophylla,Quercus_Rubra,Quercus_Semecarpifolia,Quercus_Shumardii,Quercus_Suber,Quercus_Texana,Quercus_Trojana,Quercus_Variabilis,Quercus_Vulcanica,Quercus_x_Hispanica,Quercus_x_Turneri,Rhododendron_x_Russellianum,Salix_Fragilis,Salix_Intergra,Sorbus_Aria,Tilia_Oliveri,Tilia_Platyphyllos,Tilia_Tomentosa,Ulmus_Bergmanniana,Viburnum_Tinus,Viburnum_x_Rhytidophylloides,Zelkova_Serrata'
columns = 'id,' + ','.join(classes) + '\n'
f.write(columns)

for sid, prediction in zip(s_ids, p):
    prediction = [str(v) for v in prediction]           # 컴프리헨션
    # print(sid, prediction)
    f.write('{},{}\n'.format(sid, ','.join(prediction)))

f.close()
