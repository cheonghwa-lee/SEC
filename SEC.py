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
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch

import keras
from keras import layers

def get_data(file):
    raw_data = pd.read_csv(file)
    return raw_data

def EDA(raw_data):
    print(raw_data.shape)
    print(raw_data.info())
    print(raw_data.head())
    print(raw_data.describe())

def level_1_clustering(raw_data):
    episode_num = 1
    step_num = 1
    raw_data["episode"] = 0
    raw_data["step"] = 0

    raw_data["level_1_cluster"] = -1

    for i in range(raw_data.shape[0]):
        # print("episode: ", episode_num)
        
        if (raw_data["READ"][i] <= 0.75) or (raw_data["REBD"][i] <= 0.75):
            raw_data.at[i, "episode"] = episode_num
            raw_data.at[i, "step"] = step_num
            raw_data.at[i, "level_1_cluster"] = 1 # "unsafe"
            step_num += 1

            # if (raw_data["AEX"][i] > raw_data["AEX"][i + 1]) and (0.8 < raw_data["AEX"][i + 1] < 1.2):
            if (raw_data["READ"][i+1] > 0.75) and (raw_data["REBD"][i+1] > 0.75):
                # if raw_data["AEX"][i] > 5.0:
                #     raw_data.at[i, "P/F"] = "P"
                # else:
                #     raw_data.at[i, "P/F"] = "F"
                episode_num += 1
                step_num = 1
        else:
            raw_data.at[i, "level_1_cluster"] = 0 # "safe"
            # raw_data["READ"][i] = 0
            # raw_data["REBD"][i] = 0
            raw_data.at[i, "episode"] = None
            raw_data.at[i, "step"] = None

        if i + 1 == raw_data.shape[0]:
            break

        

    # datadata = []
    # for episode in range(1, np.max(raw_data["episode"])):
    #     globals()['raw_data_{}'.format(episode)] = raw_data.loc[raw_data.episode == episode]
    #     datadata.append(globals()['raw_data_{}'.format(episode)])

    return raw_data

def level_2_clustering(raw_data):
    raw_data["level_2_cluster"] = -1

    print("1111", np.max(raw_data["episode"]), raw_data["episode"].unique())
    # print(raw_data.loc[raw_data["episode"] == 1])
    # 
    # print(np.max(tlqkf["AEX"]))
    # print("!!!!!!!!", np.max(raw_data.loc[raw_data["episode"] == 1]["AEX"]))
    for num_episode in range(1, int(np.max(raw_data["episode"]))):
        episodic_data = raw_data.loc[raw_data["episode"] == num_episode]
        if np.min(episodic_data["READ"]) < 0.21 or np.min(episodic_data["REBD"]) < 0.21:
            for i in episodic_data.index:
                # print(i)
                raw_data.at[i, "level_2_cluster"] = 1 # fail
        else:
            for i in episodic_data.index:
                # print(i)
                raw_data.at[i, "level_2_cluster"] = 0 # success

    return raw_data

    # for i in range(np.max(raw_data["episode"])):
    #     raw_data.loc[raw_data["episode"] == i]


    # pass_index = raw_data.index[raw_data["P/F"] == "P"]
    # pass_episodes = []
    # # print(pass_index)
    # for idx in pass_index:
    #     # print(idx)
    #     pass_episodes.append(raw_data["episode"][idx])
    # fail_index = raw_data.index[raw_data["P/F"] == "F"]
    # fail_episodes = []
    # for idx in fail_index:
    #     fail_episodes.append(raw_data["episode"][idx])
    # # print("P/F", pass_episodes, fail_episodes)
    # return pass_episodes, fail_episodes

def level_3_clustering(raw_data, features):
    # raw_data.loc[raw_data["level_1_cluster"] == 0]
    # raw_data.loc[raw_data["level_1_cluster"] == 1]
    # raw_data.loc[raw_data["level_2_cluster"] == 0]
    # raw_data.loc[raw_data["level_2_cluster"] == -1]

    print("2222", np.max(raw_data.loc[raw_data["level_2_cluster"] == 1]["episode"]), raw_data.loc[raw_data["level_2_cluster"] == 1]["episode"].unique())
    episode_indexes = []
    for episode in raw_data.loc[raw_data["level_2_cluster"] == 1]["episode"].unique():
        # print(raw_data.loc[raw_data["episode"] == episode].shape)
        # print(raw_data.loc[raw_data["episode"] == episode].head(raw_data.loc[raw_data["episode"] == episode].shape[0]))
        episode_first_step = raw_data.loc[raw_data["episode"] == episode][raw_data["step"] == 1].index.tolist()[0]
        # print(episode_first_step)
        episode_last_step = raw_data.loc[raw_data["episode"] == episode][raw_data["step"] == raw_data.loc[raw_data["episode"] == episode].shape[0]].index.tolist()[0]
        # print(episode_last_step)
        # print(episode_last_step - episode_first_step)
        indexes = []
        indexes_1 = []
        indexes_2 = []
        indexes_3 = []
        num = 10
        if raw_data.loc[raw_data["episode"] == episode].shape[0] >= num:
            indexes_1.append(episode_first_step)
            for index in range(episode_first_step, episode_last_step, int(round((episode_last_step-episode_first_step)/(num-1)))):
                indexes_1.append(index)
                # print(int((episode_last_step-episode_first_step)/(num-1)))
            indexes_1.append(episode_last_step)
            # print(np.array(indexes_1).shape)
            indexes_2.append(episode_first_step)
            for index in range(episode_first_step, episode_last_step, int(round((episode_last_step-episode_first_step)/(num-2)))):
                indexes_2.append(index)
                # print(int((episode_last_step-episode_first_step)/(num-1)))
            indexes_2.append(episode_last_step)
            # print(np.array(indexes_2).shape)
            indexes_3.append(episode_first_step)
            for index in range(episode_first_step, episode_last_step, int(round((episode_last_step-episode_first_step)/(num-3)))):
                indexes_3.append(index)
                # print(int((episode_last_step-episode_first_step)/(num-1)))
            indexes_3.append(episode_last_step)
            # print(np.array(indexes_3).shape)
            if np.array(indexes_1).shape[0] == 10:
                indexes = indexes_1
            elif np.array(indexes_2).shape[0] == 10:
                indexes = indexes_2
            elif np.array(indexes_3).shape[0] == 10:
                indexes = indexes_3
            elif np.array(indexes_3).shape[0] == 9:
                indexes = indexes_3
                indexes.append(episode_last_step)
            elif np.array(indexes_3).shape[0] == 8:
                indexes = indexes_3
                indexes.append(episode_last_step)
                indexes.append(episode_last_step)
            elif np.array(indexes_3).shape[0] > 10:
                indexes = indexes_3[1:]
            else:
                print("error")
            # print(np.array(indexes).shape)
            # print(indexes)
            episode_indexes.append(indexes)
    print("tlqkf", np.array(episode_indexes).shape)
    # figure_cartesian_episode(1004, raw_data, episode_indexes)

    clustering_data = []
    visualization_data = []
    vfeatures = ["AEX", "AEY", "AEZ", "AAX", "AAY", "AAZ", "ABX", "ABY", "ABZ"]
    for episode_index in episode_indexes:
        # episode_data = []
        cdata = []
        vdata = []
        for index in episode_index:
            # data = []
            for feature in features:
                cdata.append(raw_data[feature][index])
            # print("!@#$", np.array(data).shape)
            # episode_data.append(data)
            for vfeature in vfeatures:
                vdata.append(raw_data[vfeature][index])
        # print("!@#$", np.array(episode_data).shape)
        # clustering_data.append(episode_data)
        clustering_data.append(cdata)
        visualization_data.append(vdata)
    # for idx in range(304):
        # print("!@#$", np.array(clustering_data[idx]).shape)
    
    print("!@#$", np.array(clustering_data).shape)

    CLUSTER = 5

    model = KMeans(n_clusters=CLUSTER, random_state=10).fit(clustering_data)
    dic1 = {name:value for name, value in zip(model.labels_, visualization_data)}
    figure_clustering_algorithm(6, dic1)

    model777 = KMeans(n_clusters=CLUSTER, random_state=10).fit(visualization_data)
    # value1 = model.cluster_centers_
    # dic1 = {name:value for name, value in zip(model.labels_, visualization_data)}
    dic777 = {name:value for name, value in zip([i for i in range(CLUSTER)], model777.cluster_centers_.tolist())}
    # print("tlqjltjlwjtlkjwjtlkwjtlkwjtlkjwtlkwjtl", dic1)
    figure_clustering_algorithm(6777, dic777)
    elbow(777, clustering_data)

    clustering_relative = AffinityPropagation(random_state=5).fit(clustering_data)
    dic2 = {name:value for name, value in zip(clustering_relative.labels_, visualization_data)} # do not touch!
    # figure_clustering_algorithm(7, dic2)

    MS_clustering_relative = MeanShift(bandwidth=0.2).fit(clustering_data)
    dic3 = {name:value for name, value in zip(MS_clustering_relative.labels_, visualization_data)} # do not touch!
    figure_clustering_algorithm(8, dic3)

    clustering_relative = SpectralClustering(n_clusters=CLUSTER, assign_labels='discretize', random_state=10).fit(clustering_data)
    dic4 = {name:value for name, value in zip(clustering_relative.labels_, visualization_data)} # do not touch!
    figure_clustering_algorithm(9, dic4)

    clustering_relative = AgglomerativeClustering(n_clusters=CLUSTER).fit(clustering_data)
    dic5 = {name:value for name, value in zip(clustering_relative.labels_, visualization_data)} # do not touch!
    figure_clustering_algorithm(10, dic5)

    DBSCAN_clustering_relative = DBSCAN(eps=0.5, min_samples=CLUSTER).fit(clustering_data)
    dic6 = {name:value for name, value in zip(DBSCAN_clustering_relative.labels_, visualization_data)} # do not touch!
    # figure_clustering_algorithm(11, dic6)

    clustering_relative = OPTICS(min_samples=CLUSTER).fit(clustering_data)
    dic7 = {name:value for name, value in zip(clustering_relative.labels_, visualization_data)} # do not touch!
    # figure_clustering_algorithm(12, dic7)

    GMM_clustering_relative = GaussianMixture(n_components=CLUSTER, random_state=10).fit(clustering_data)
    labels = GMM_clustering_relative.fit_predict(clustering_data)
    dic8 = {name:value for name, value in zip(labels, visualization_data)} # do not touch!
    figure_clustering_algorithm(13, dic8)

    clustering_relative = Birch(n_clusters=CLUSTER).fit(clustering_data)
    dic9 = {name:value for name, value in zip(clustering_relative.labels_, visualization_data)} # do not touch!
    figure_clustering_algorithm(14, dic9)
    
    # # random plot
    # for j in range(10):
    #     figure_cartesian_random(1000+j, visualization_data)
    # # for j in range(10):
    # #     figure_cartesian_random(2000+j, fail_visualization_data)
    
    # figure_cartesian_3d(100, visualization_data)
    # # figure_cartesian_2d(101, clustering_data)

    # # figure_cartesian_3d(102, fail_visualization_data)
    # # figure_cartesian_2d(103, fail_clustering_data)

    # # level_2_clustering(raw_data, episodic_data)

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
    raw_data = get_data("episode_data_01_53_02.csv")
    # EDA(raw_data)
    # figure_timeseries(1, raw_data, raw_data.shape[0])

    raw_data = level_1_clustering(raw_data)
    print(raw_data.loc[raw_data["level_1_cluster"] == 1].shape)
    print(raw_data.loc[raw_data["level_1_cluster"] == 0].shape)
    print(raw_data.loc[raw_data["level_1_cluster"] == -1].shape)
    # EDA(raw_data)
    # figure_timeseries(2, raw_data, raw_data.shape[0])

    raw_data = level_2_clustering(raw_data)
    # EDA(raw_data)
    print(raw_data.loc[raw_data["level_2_cluster"] == 1].shape)
    print(raw_data.loc[raw_data["level_2_cluster"] == 0].shape)
    print(raw_data.loc[raw_data["level_2_cluster"] == -1].shape)

    # figure_timeseries_episodic_start_end(2, episodic_data, 250, 255)
    # figure_cartesian(3, episodic_data)
    # figure_cartesian_start_end(4, episodic_data, 250, 255)

    features = ["REAX", "REAY", "REAZ", "REBX", "REBY", "REBZ", "READ", "REBD"]
    # features = ["REAX", "REAY", "REAZ", "REBX", "REBY", "REBZ"]
    # features = ["REAX", "REAY", "REBX", "REBY"]
    # features = ["AEX", "AEY", "AEZ", "AAX", "AAY", "AAZ", "ABX", "ABY", "ABZ", "REAX", "REAY", "REAZ", "REBX", "REBY", "REBZ"]
    # features = ["AEX", "AEY", "AEZ", "AAX", "AAY", "AAZ", "ABX", "ABY", "ABZ"]
    # features = ["AEX", "AEY", "AAX", "AAY", "ABX", "ABY", "REAX", "REAY", "REBX", "REBY"]
    # features = ["REAX", "REAY", "REBX", "REBY", "READ", "REBD"]
    level_3_clustering(raw_data, features)

    

    CLUSTER = 3

    # condition1 = create_data_()
    # # print(condition1)
    # figure_cartesian_single(888, condition1)

    # model = KMeans(n_clusters=CLUSTER, random_state=20).fit(condition1)
    # dic111 = {name:value for name, value in zip([i for i in range(CLUSTER)], model.cluster_centers_.tolist())}
    # # dic112 = {name:value for name, value in zip([i for i in range(CLUSTER)], condition1)}
    # figure_clustering_algorithm(665, dic111)
    # # figure_clustering_algorithm(666, dic112)
    # elbow(667, condition1)

    # # clustering_relative = AffinityPropagation(random_state=5).fit(condition1)
    # # # dic2 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # # dic2 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.cluster_centers_.tolist())}
    # # figure_clustering_algorithm(7, dic2)

    # MS_clustering_relative = MeanShift(bandwidth=0.2).fit(condition1)
    # # dic3 = {name:value for name, value in zip(MS_clustering_relative.labels_, condition1)} # do not touch!
    # dic3 = {name:value for name, value in zip([i for i in range(CLUSTER)], MS_clustering_relative.cluster_centers_.tolist())}
    # figure_clustering_algorithm(8, dic3)

    # # clustering_relative = SpectralClustering(n_clusters=CLUSTER, assign_labels='discretize', random_state=10).fit(condition1)
    # # # dic4 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # # dic4 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.affinity_matrix_.tolist())}
    # # figure_clustering_algorithm(9, dic4)

    # clustering_relative = AgglomerativeClustering(n_clusters=CLUSTER).fit(condition1)
    # dic5 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # # dic5 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.distances_.tolist())}
    # figure_clustering_algorithm(10, dic5)

    # # DBSCAN_clustering_relative = DBSCAN(eps=0.5, min_samples=CLUSTER).fit(condition1)
    # # # dic6 = {name:value for name, value in zip(DBSCAN_clustering_relative.labels_, condition1)} # do not touch!
    # # dic6 = {name:value for name, value in zip([i for i in range(CLUSTER)], DBSCAN_clustering_relative.components_.tolist())}
    # # figure_clustering_algorithm(11, dic6)

    # # clustering_relative = OPTICS(min_samples=CLUSTER).fit(condition1)
    # # # dic7 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # # dic7 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.core_distances_.tolist())}
    # # # figure_clustering_algorithm(12, dic7)

    # GMM_clustering_relative = GaussianMixture(n_components=CLUSTER, random_state=10).fit(condition1)
    # labels = GMM_clustering_relative.fit_predict(condition1)
    # dic8 = {name:value for name, value in zip(labels, condition1)} # do not touch!
    # # dic8 = {name:value for name, value in zip([i for i in range(CLUSTER)], GMM_clustering_relative.means_.tolist())}
    # figure_clustering_algorithm(13, dic8)

    # clustering_relative = Birch(n_clusters=CLUSTER).fit(condition1)
    # # dic9 = {name:value for name, value in zip(clustering_relative.labels_, condition1)} # do not touch!
    # dic9 = {name:value for name, value in zip([i for i in range(CLUSTER)], clustering_relative.subcluster_centers_.tolist())}
    # figure_clustering_algorithm(14, dic9)

    plt.show()

if __name__ == "__main__":
    main()