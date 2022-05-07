# from curses import raw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import random

from Figure_SEC import *
from Clustering_SEC import *

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
# from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch

import keras
from keras import layers

def get_data(file):
    raw_data = pd.read_csv(file)
    return raw_data

# def EDA(raw_data):
#     print(raw_data.shape)
#     print(raw_data.info())
#     print(raw_data.head())
#     print(raw_data.describe())

# def create_data():
#     conditions = []
#     for _ in range(5):
#         xerror = 0.4
#         yerror = 0.1
#         zerror = -3.14 / 20
#         AEX = 1.0 + random.uniform(-xerror, xerror)
#         AEY = 0.15 + random.uniform(-yerror, yerror)
#         AEZ = 0.0 + random.uniform(-zerror, zerror)
#         AAX = 0.0 + random.uniform(-xerror, xerror)
#         AAY = -0.15 + random.uniform(-yerror, yerror)
#         AAZ = 0.0 + random.uniform(-zerror, zerror)
#         ABX = 2.0 + random.uniform(-xerror, xerror)
#         ABY = 0.15 + random.uniform(-yerror, yerror)
#         ABZ = 0.0 + random.uniform(-zerror, zerror)
#         # AEX = 1.0 
#         # AEY = 0.15 
#         # AEZ = 0.0
#         # AAX = 0.0 
#         # AAY = -0.15 
#         # AAZ = 0.0 
#         # ABX = 2.0 
#         # ABY = 0.15 
#         # ABZ = 0.0
#         conditions.append([AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ])
        
#         AEX = 0.0 + random.uniform(-xerror, xerror)
#         AEY = -0.15 + random.uniform(-yerror, yerror)
#         AEZ = 0.0 + random.uniform(-zerror, zerror)
#         AAX = 2.0 + random.uniform(-xerror, xerror)
#         AAY = -0.15 + random.uniform(-yerror, yerror)
#         AAZ = 0.0 + random.uniform(-zerror, zerror)
#         ABX = 5.0 + random.uniform(-xerror, xerror)
#         ABY = 0.15 + random.uniform(-yerror, yerror)
#         ABZ = -3.14 + random.uniform(-zerror, zerror)
#         conditions.append([AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ])
        
#         AEX = 1.0 + random.uniform(-xerror, xerror)
#         AEY = 0.15 + random.uniform(-yerror, yerror)
#         AEZ = 0.0 + random.uniform(-zerror, zerror)
#         AAX = 3.0 + random.uniform(-xerror, xerror)
#         AAY = 0.5 + random.uniform(-yerror, yerror)
#         AAZ = -3.14 / 2 + random.uniform(-zerror, zerror)
#         ABX = 5.0 + random.uniform(-xerror, xerror)
#         ABY = -0.5 + random.uniform(-yerror, yerror)
#         ABZ = 3.14 / 2 + random.uniform(-zerror, zerror)
#         conditions.append([AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ])

#     return conditions

def create_data_():
    conditions = []
    range_ = 15
    for _ in range(5):
        xerror = 0.4
        yerror = 0.1
        zerror = -3.14 / 20
        tlqkf1 = []
        AEX = 1.0 
        AEY = 0.15 
        AEZ = 0.0 
        AAX = 0.0 
        AAY = -0.15 
        AAZ = 0.0 
        ABX = 2.0 
        ABY = 0.15 
        ABZ = 0.0 
        for _ in range(range_):
            AEX += 0.2
            AEY += 0
            AEZ += 0 + random.uniform(-zerror, zerror)
            AAX += 0.3
            AAY += 0
            AAZ += 0 + random.uniform(-zerror, zerror)
            ABX += 0.1
            ABY += 0
            ABZ += 0 + random.uniform(-zerror, zerror)
            # AEX = 1.0 
            # AEY = 0.15 
            # AEZ = 0.0
            # AAX = 0.0 
            # AAY = -0.15 
            # AAZ = 0.0 
            # ABX = 2.0 
            # ABY = 0.15 
            # ABZ = 0.0
            tlqkf1 += [AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ]
        conditions.append(tlqkf1)
        
        tlqkf2 = []
        AEX = 1.0 
        AEY = 0.15 
        AEZ = 0.0 
        AAX = 0.0 
        AAY = -0.15 
        AAZ = 0.0 
        ABX = 2.0 
        ABY = 0.15 
        ABZ = 0.0 
        for _ in range(range_):
            AEX += 0.3
            AEY += 0
            AEZ += 0 + random.uniform(-zerror, zerror)
            AAX += 0.1
            AAY += 0
            AAZ += 0 + random.uniform(-zerror, zerror)
            ABX += 0.4
            ABY += 0
            ABZ += 0 + random.uniform(-zerror, zerror)
            # AEX = 1.0 
            # AEY = 0.15 
            # AEZ = 0.0
            # AAX = 0.0 
            # AAY = -0.15 
            # AAZ = 0.0 
            # ABX = 2.0 
            # ABY = 0.15 
            # ABZ = 0.0
            tlqkf2 += [AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ]
        conditions.append(tlqkf2)

        tlqkf3 = []
        AEX = 1.0 
        AEY = 0.15 
        AEZ = 0.0 
        AAX = 0.0 
        AAY = -0.15 
        AAZ = 0.0 
        ABX = 2.0 
        ABY = 0.15 
        ABZ = 0.0 
        for _ in range(range_):
            AEX += 0.0
            AEY += 0
            AEZ += 0 + random.uniform(-zerror, zerror)
            AAX += 0.1
            AAY += 0
            AAZ += 0 + random.uniform(-zerror, zerror)
            ABX += 0.2
            ABY += 0
            ABZ += 0 + 0.1
            # AEX = 1.0 
            # AEY = 0.15 
            # AEZ = 0.0
            # AAX = 0.0 
            # AAY = -0.15 
            # AAZ = 0.0 
            # ABX = 2.0 
            # ABY = 0.15 
            # ABZ = 0.0
            tlqkf3 += [AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ]
        conditions.append(tlqkf3)

        # tlqkf2 = []
        # AEX = 0.0 
        # AEY = -0.15 
        # AEZ = 0.0 
        # AAX = 2.0 
        # AAY = -0.15 
        # AAZ = 0.0 
        # ABX = 5.0 
        # ABY = 0.15 
        # ABZ = -3.14 
        # for _ in range(range_):
        #     AEX += 0.1
        #     AEY += 0
        #     AEZ += 0 + random.uniform(-zerror, zerror)
        #     AAX += 0.3
        #     AAY += 0
        #     AAZ += 0 + random.uniform(-zerror, zerror)
        #     ABX += -0.3
        #     ABY += 0
        #     ABZ += 0 + random.uniform(-zerror, zerror)
        #     tlqkf2 += [AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ]
        # conditions.append(tlqkf2)
        
        # tlqkf3 = []
        # AEX = 1.0 
        # AEY = 0.15 
        # AEZ = 0.0 
        # AAX = 3.0 
        # AAY = 0.5 
        # AAZ = -3.14 / 2 
        # ABX = 5.0 
        # ABY = -0.5 
        # ABZ = 3.14 / 2 
        # for _ in range(range_):
        #     AEX += 0.2
        #     AEY += 0
        #     AEZ += 0 + random.uniform(-zerror, zerror)
        #     AAX += 0
        #     AAY += -0.2
        #     AAZ += 0 + random.uniform(-zerror, zerror)
        #     ABX += 0
        #     ABY += 0.2
        #     ABZ += 0 + random.uniform(-zerror, zerror)
        #     tlqkf3 += [AEX, AEY, AEZ, AAX, AAY, AAZ, ABX, ABY, ABZ]
        # print("==================================================", np.array(tlqkf3).shape)
        conditions.append(tlqkf3)

    print("22-05-06", np.array(conditions).shape)

    return np.array(conditions)

def detect_rc(raw_data):
    episode_num = 1
    step_num = 1
    raw_data["episode"] = 0
    raw_data["step"] = 0
    for i in range(raw_data.shape[0]):
        raw_data.at[i, "episode"] = episode_num
        raw_data.at[i, "step"] = step_num
        step_num += 1
        if i + 1 == raw_data.shape[0]:
            break
        if (raw_data["AEX"][i] > raw_data["AEX"][i + 1]) and (0.9 < raw_data["AEX"][i + 1] < 1.1):
            if raw_data["AEX"][i] > 5.0:
                raw_data.at[i, "P/F"] = "P"
            else:
                raw_data.at[i, "P/F"] = "F"
            episode_num += 1
            step_num = 1

    datadata = []
    for episode in range(1, np.max(raw_data["episode"])):
        globals()['raw_data_{}'.format(episode)] = raw_data.loc[raw_data.episode == episode]
        datadata.append(globals()['raw_data_{}'.format(episode)])

    return datadata

def pass_fail_data(raw_data):
    pass_index = raw_data.index[raw_data["P/F"] == "P"]
    pass_episodes = []
    # print(pass_index)
    for idx in pass_index:
        # print(idx)
        pass_episodes.append(raw_data["episode"][idx])
    fail_index = raw_data.index[raw_data["P/F"] == "F"]
    fail_episodes = []
    for idx in fail_index:
        fail_episodes.append(raw_data["episode"][idx])
    # print("P/F", pass_episodes, fail_episodes)
    return pass_episodes, fail_episodes

def clustering_data(datadata, features, pass_episodes, fail_episodes):
    start_f = 0
    end_f = 300
    pass_conditions = []
    pass_conditions_rel = []
    fail_conditions = []
    fail_conditions_rel = []
    for rd in datadata[start_f: end_f]:
        init_c = rd.index[rd["step"] == 1]
        finish_c = rd.index[rd["step"] == len(rd)-1] 
        # print(rd["episode"].tolist()[0], pass_episodes)
        if rd["episode"].tolist()[0] in pass_episodes:
            # pass_conditions, pass_conditions_rel = tlqkf(rd, init_c, features, pass_conditions, pass_conditions_rel)
            pass_conditions, pass_conditions_rel = tlqkf(rd, finish_c, features, pass_conditions, pass_conditions_rel)
        elif rd["episode"].tolist()[0] in fail_episodes:
            # fail_conditions, fail_conditions_rel = tlqkf(rd, init_c, features, fail_conditions, fail_conditions_rel)
            fail_conditions, fail_conditions_rel = tlqkf(rd, finish_c, features, fail_conditions, fail_conditions_rel)
        else:
            print("ERROR!!!!!!!!")
    pass_conditions = np.array(pass_conditions)
    pass_conditions = pass_conditions.squeeze()
    fail_conditions = np.array(fail_conditions)
    fail_conditions = fail_conditions.squeeze()
    return pass_conditions, pass_conditions_rel, fail_conditions, fail_conditions_rel

def tlqkf(rd, finish_c, features, conditions, conditions_rel):
    # conditions = []
    # conditions_rel = []
    # condition_raw = []
    
    condition = [rd["AEX"][finish_c].tolist() + rd["AEY"][finish_c].tolist() + rd["AEZ"][finish_c].tolist() + rd["AAX"][finish_c].tolist() + rd["AAY"][finish_c].tolist() + rd["AAZ"][finish_c].tolist() + rd["ABX"][finish_c].tolist() + rd["ABY"][finish_c].tolist() + rd["ABZ"][finish_c].tolist()]
    condition_raw = []
    # condition = []
    for idx in features:
        condition_raw.append(np.array(rd[idx][finish_c].tolist()).squeeze())
    conditions_rel.append(condition_raw)
    conditions.append(condition)

    
    return conditions, conditions_rel

def main():
    raw_data = get_data("episode_data_11_08_24.csv")
    # figure_timeseries(1, raw_data, 1000)
    # starts = [10000]
    # ends = [11000]
    # figure_timeseries_1000(333, raw_data, starts, ends)
    # episodic_data = detect_rc(raw_data)
    # pass_episodes, fail_episodes = pass_fail_data(raw_data)
    # # figure_timeseries_episodic_start_end(2, episodic_data, 250, 255)
    # # figure_cartesian(3, episodic_data)
    # figure_cartesian_start_end(4, episodic_data, 250, 255)
    # features = ["REAX", "REAY", "REAZ", "REBX", "REBY", "REBZ", "READ", "REBD"]
    # # features = ["REAX", "REAY", "REAZ", "REBX", "REBY", "REBZ"]
    # # features = ["REAX", "REAY", "REBX", "REBY"]
    # # features = ["AEX", "AEY", "AEZ", "AAX", "AAY", "AAZ", "ABX", "ABY", "ABZ", "REAX", "REAY", "REAZ", "REBX", "REBY", "REBZ"]
    # # features = ["AEX", "AEY", "AEZ", "AAX", "AAY", "AAZ", "ABX", "ABY", "ABZ"]
    # # features = ["AEX", "AEY", "AAX", "AAY", "ABX", "ABY", "REAX", "REAY", "REBX", "REBY"]
    # # features = ["REAX", "REAY", "REBX", "REBY", "READ", "REBD"]
    # pass_conditions_abs, pass_conditions_rel, conditions_abs, conditions_rel = clustering_data(episodic_data, features, pass_episodes, fail_episodes)

    # CLUSTER = 4

    # model = KMeans(n_clusters=CLUSTER, random_state=10).fit(conditions_rel)
    # dic1 = {name:value for name, value in zip(model.labels_, conditions_abs)}
    # figure_clustering_algorithm(6, dic1)

    # model777 = KMeans(n_clusters=CLUSTER, random_state=10).fit(conditions_abs)
    # # value1 = model.cluster_centers_
    # # dic1 = {name:value for name, value in zip(model.labels_, conditions_abs)}
    # dic777 = {name:value for name, value in zip([i for i in range(CLUSTER)], model777.cluster_centers_.tolist())}
    # print("tlqjltjlwjtlkjwjtlkwjtlkwjtlkjwtlkwjtl", dic1)
    # figure_clustering_algorithm(6777, dic777)
    # elbow(777, conditions_abs)

    # clustering_relative = AffinityPropagation(random_state=5).fit(conditions_rel)
    # dic2 = {name:value for name, value in zip(clustering_relative.labels_, conditions_abs)} # do not touch!
    # # figure_clustering_algorithm(7, dic2)

    # MS_clustering_relative = MeanShift(bandwidth=0.2).fit(conditions_rel)
    # dic3 = {name:value for name, value in zip(MS_clustering_relative.labels_, conditions_abs)} # do not touch!
    # figure_clustering_algorithm(8, dic3)

    # clustering_relative = SpectralClustering(n_clusters=CLUSTER, assign_labels='discretize', random_state=10).fit(conditions_rel)
    # dic4 = {name:value for name, value in zip(clustering_relative.labels_, conditions_abs)} # do not touch!
    # figure_clustering_algorithm(9, dic4)

    # clustering_relative = AgglomerativeClustering(n_clusters=CLUSTER).fit(conditions_rel)
    # dic5 = {name:value for name, value in zip(clustering_relative.labels_, conditions_abs)} # do not touch!
    # figure_clustering_algorithm(10, dic5)

    # DBSCAN_clustering_relative = DBSCAN(eps=0.5, min_samples=CLUSTER).fit(conditions_rel)
    # dic6 = {name:value for name, value in zip(DBSCAN_clustering_relative.labels_, conditions_abs)} # do not touch!
    # # figure_clustering_algorithm(11, dic6)

    # clustering_relative = OPTICS(min_samples=CLUSTER).fit(conditions_rel)
    # dic7 = {name:value for name, value in zip(clustering_relative.labels_, conditions_abs)} # do not touch!
    # # figure_clustering_algorithm(12, dic7)

    # GMM_clustering_relative = GaussianMixture(n_components=CLUSTER, random_state=10).fit(conditions_rel)
    # labels = GMM_clustering_relative.fit_predict(conditions_rel)
    # dic8 = {name:value for name, value in zip(labels, conditions_abs)} # do not touch!
    # figure_clustering_algorithm(13, dic8)

    # clustering_relative = Birch(n_clusters=CLUSTER).fit(conditions_rel)
    # dic9 = {name:value for name, value in zip(clustering_relative.labels_, conditions_abs)} # do not touch!
    # figure_clustering_algorithm(14, dic9)
    
    # # random plot
    # for j in range(10):
    #     figure_cartesian_random(1000+j, conditions_abs)
    # # for j in range(10):
    # #     figure_cartesian_random(2000+j, fail_conditions_abs)
    
    # figure_cartesian_3d(100, conditions_abs)
    # # figure_cartesian_2d(101, conditions_rel)

    # # figure_cartesian_3d(102, fail_conditions_abs)
    # # figure_cartesian_2d(103, fail_conditions_rel)

    # # pass_fail_data(raw_data, episodic_data)

    CLUSTER = 3

    condition1 = create_data_()
    # print(condition1)
    figure_cartesian_single(888, condition1)

    model = KMeans(n_clusters=CLUSTER, random_state=20).fit(condition1)
    dic111 = {name:value for name, value in zip([i for i in range(CLUSTER)], model.cluster_centers_.tolist())}
    # dic112 = {name:value for name, value in zip([i for i in range(CLUSTER)], condition1)}
    figure_clustering_algorithm(665, dic111)
    # figure_clustering_algorithm(666, dic112)
    elbow(667, condition1)

    # clustering_relative = AffinityPropagation(random_state=5).fit(condition1)
    # # dic2 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # dic2 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.cluster_centers_.tolist())}
    # figure_clustering_algorithm(7, dic2)

    MS_clustering_relative = MeanShift(bandwidth=0.2).fit(condition1)
    # dic3 = {name:value for name, value in zip(MS_clustering_relative.labels_, condition1)} # do not touch!
    dic3 = {name:value for name, value in zip([i for i in range(CLUSTER)], MS_clustering_relative.cluster_centers_.tolist())}
    figure_clustering_algorithm(8, dic3)

    # clustering_relative = SpectralClustering(n_clusters=CLUSTER, assign_labels='discretize', random_state=10).fit(condition1)
    # # dic4 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # dic4 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.affinity_matrix_.tolist())}
    # figure_clustering_algorithm(9, dic4)

    clustering_relative = AgglomerativeClustering(n_clusters=CLUSTER).fit(condition1)
    dic5 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # dic5 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.distances_.tolist())}
    figure_clustering_algorithm(10, dic5)

    # DBSCAN_clustering_relative = DBSCAN(eps=0.5, min_samples=CLUSTER).fit(condition1)
    # # dic6 = {name:value for name, value in zip(DBSCAN_clustering_relative.labels_, condition1)} # do not touch!
    # dic6 = {name:value for name, value in zip([i for i in range(CLUSTER)], DBSCAN_clustering_relative.components_.tolist())}
    # figure_clustering_algorithm(11, dic6)

    # clustering_relative = OPTICS(min_samples=CLUSTER).fit(condition1)
    # # dic7 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # dic7 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.core_distances_.tolist())}
    # # figure_clustering_algorithm(12, dic7)

    GMM_clustering_relative = GaussianMixture(n_components=CLUSTER, random_state=10).fit(condition1)
    labels = GMM_clustering_relative.fit_predict(condition1)
    dic8 = {name:value for name, value in zip(labels, condition1)} # do not touch!
    # dic8 = {name:value for name, value in zip([i for i in range(CLUSTER)], GMM_clustering_relative.means_.tolist())}
    figure_clustering_algorithm(13, dic8)

    clustering_relative = Birch(n_clusters=CLUSTER).fit(condition1)
    # dic9 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    dic9 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.subcluster_centers_.tolist())}
    figure_clustering_algorithm(14, dic9)















    # # This is the size of our encoded representations
    # encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # # This is our input image
    # input_img = keras.Input(shape=(135,))
    # # "encoded" is the encoded representation of the input
    # encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = layers.Dense(135, activation='sigmoid')(encoded)

    # # This model maps an input to its reconstruction
    # autoencoder = keras.Model(input_img, decoded)

    # # This model maps an input to its encoded representation
    # encoder = keras.Model(input_img, encoded)

    # # This is our encoded (32-dimensional) input
    # encoded_input = keras.Input(shape=(encoding_dim,))
    # # Retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # Create the decoder model
    # decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # # from keras.datasets import mnist
    # # import numpy as np
    # # (x_train, _), (x_test, _) = mnist.load_data()

    # # x_train = x_train.astype('float32') / 255.
    # # x_test = x_test.astype('float32') / 255.
    # # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # # print(x_train.shape)
    # # print(x_test.shape)

    # # autoencoder.fit(x_train, x_train,
    # #                 epochs=50,
    # #                 batch_size=256,
    # #                 shuffle=True,
    # #                 validation_data=(x_test, x_test))
    # autoencoder.fit(condition1, condition1,
    #                 epochs=500,
    #                 batch_size=5,
    #                 shuffle=True)

    # # Encode and decode some digits
    # # Note that we take them from the *test* set
    # encoded_imgs = encoder.predict(condition1)
    # decoded_imgs = decoder.predict(encoded_imgs)

    # # Use Matplotlib (don't ask)
    # import matplotlib.pyplot as plt

    # n = 10  # How many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # Display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(condition1[i].reshape(15, 9))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # Display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded_imgs[i].reshape(15, 9))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

















    plt.show()

if __name__ == "__main__":
    main()