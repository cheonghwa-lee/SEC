import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

def figure_timeseries(i, raw_data, end):
    # raw_data.plot()
    INDEX = 7
    start = 0
    # end = raw_data.shape[0]
    end = end
    term = end - start
    plt.figure(i)
    plt.xlim([start, end])
    # plt.ylabel("Physical Parameters")
    # plt.xlabel("Step")

    plt.subplot(INDEX, 1, 1)
    plt.ylabel("x")
    plt.grid(True)
    plt.ylim([0, 5])
    plt.plot(raw_data["AEX"][start:end], "g", label='E')
    plt.plot(raw_data["AAX"][start:end], "b", label='A')
    plt.plot(raw_data["ABX"][start:end], "r", label='B')


    plt.subplot(INDEX, 1, 2)
    plt.ylabel("y")
    plt.grid(True)
    plt.plot(raw_data["REAX"][start:end], "b")
    plt.plot(raw_data["REBX"][start:end], "r")
    plt.plot([0.21]*term, "k--", label="Ref.")
    plt.plot([-0.21]*term, "k--")

    plt.subplot(INDEX, 1, 3)
    plt.ylabel("$\phi$")
    plt.grid(True)
    plt.plot(raw_data["AEY"][start:end], "g")
    plt.plot(raw_data["AAY"][start:end], "b")
    plt.plot(raw_data["ABY"][start:end], "r")

    plt.subplot(INDEX, 1, 4)
    plt.ylabel("$\Delta$x")
    plt.grid(True)
    plt.plot(raw_data["REAY"][start:end], "b")
    plt.plot(raw_data["REBY"][start:end], "r")
    plt.plot([0.21]*term, "k--")
    plt.plot([-0.21]*term, "k--")

    plt.subplot(INDEX, 1, 5)
    plt.ylabel("$\Delta$y")
    plt.grid(True)
    plt.plot(np.degrees(raw_data["AEZ"][start:end]), "g")
    plt.plot(np.degrees(raw_data["AAZ"][start:end]), "b")
    plt.plot(np.degrees(raw_data["ABZ"][start:end]), "r")

    plt.subplot(INDEX, 1, 6)
    plt.ylabel("$\Delta\phi$")
    plt.grid(True)
    plt.plot(np.degrees(raw_data["REAZ"][start:end]), "b")
    plt.plot(np.degrees(raw_data["REBZ"][start:end]), "r")

    plt.subplot(INDEX, 1, 7)
    plt.ylabel("d")
    plt.grid(True)
    plt.plot([0.21]*term, "k--")
    plt.plot([1.00]*term, "k--")
    plt.plot(raw_data["READ"][start:end], "b")
    plt.plot(raw_data["REBD"][start:end], "r")

def figure_cartesian_episode(i, raw_data, episode_indexes):
    plt.figure(i)
    # plt.figure(figsize=(12, 3))
    # plt.xlim([-0.5, 5.5])
    # plt.ylim([-0.5, 0.5])
    # plt.axis('scaled')
    ii = 1
    maxfigure = np.array(episode_indexes).shape[0]
    print("maxfigure: ", maxfigure)
    # conditions = []
    for episode_index in episode_indexes:
        # index = rd.index[rd["step"] == 1]
        # finish_c = rd.index[rd["step"] == len(rd)-1] # 
        magnitude = 0.21
        plt.subplot(maxfigure, 1, ii)
        plt.axis('scaled')
        plt.xlim([-0.5, 5.5])
        plt.ylim([-0.5, 0.5])
        for index in episode_index:
            plt.plot(raw_data["AEX"][index], raw_data["AEY"][index], "g.")
            circle1 = plt.Circle((raw_data["AEX"][index], raw_data["AEY"][index]), 0.105, fill=False)
            plt.gca().add_patch(circle1)
            x1=magnitude*np.cos(raw_data["AEZ"][index])
            y1=magnitude*np.sin(raw_data["AEZ"][index])
            plt.annotate("", xy=(x1+raw_data["AEX"][index], y1+raw_data["AEY"][index]), xytext=(raw_data["AEX"][index], raw_data["AEY"][index]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

            plt.plot(raw_data["AAX"][index], raw_data["AAY"][index], "b.")
            circle2 = plt.Circle((raw_data["AAX"][index], raw_data["AAY"][index]), 0.105, fill=False)
            plt.gca().add_patch(circle2)
            x2=magnitude*np.cos(raw_data["AAZ"][index])
            y2=magnitude*np.sin(raw_data["AAZ"][index])
            plt.annotate("", xy=(x2+raw_data["AAX"][index], y2+raw_data["AAY"][index]), xytext=(raw_data["AAX"][index], raw_data["AAY"][index]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

            plt.plot(raw_data["ABX"][index], raw_data["ABY"][index], "r.")
            circle3 = plt.Circle((raw_data["ABX"][index], raw_data["ABY"][index]), 0.105, fill=False)
            plt.gca().add_patch(circle3)
            x3=magnitude*np.cos(raw_data["ABZ"][index])
            y3=magnitude*np.sin(raw_data["ABZ"][index])
            plt.annotate("", xy=(x3+raw_data["ABX"][index], y3+raw_data["ABY"][index]), xytext=(raw_data["ABX"][index], raw_data["ABY"][index]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})
        ii += 1
        # conditions = [rd["AEX"][finish_c], rd["AEY"][finish_c], rd["AEZ"][finish_c], rd["AAX"][finish_c], rd["AAY"][finish_c], rd["AAZ"][finish_c], rd["ABX"][finish_c], rd["ABY"][finish_c], rd["ABZ"][finish_c]]

def figure_clustering_algorithm(i, dic):
    plt.figure(i)
    # plt.xlim([-1, 6])
    # plt.ylim([-0.3, 0.3])
    # print("ttttttttttttt", len(dic[0]))
    for key in dic.keys():
        value = dic[key]
        # print(len(value))
        # finish_c = rd.index[rd["step"] == len(rd)-1] # 
        magnitude = 0.21
        feature = 9
        range_ = int(len(value) / feature)
        for scene in range(range_):
            plt.subplot(len(dic), 1, key +1)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis('scaled')
            plt.xlim([-1, 6])
            # plt.ylim([-0.3, 0.3])
            plt.ylim([-0.75, 0.75])
            plt.plot(value[0+scene*feature], value[1+scene*feature], "g.")
            circle1 = plt.Circle((value[0+scene*feature], value[1+scene*feature]), 0.105, fill=False)
            plt.gca().add_patch(circle1)
            x1=magnitude*np.cos(value[2+scene*feature])
            y1=magnitude*np.sin(value[2+scene*feature])
            plt.annotate("", xy=(x1+value[0+scene*feature], y1+value[1+scene*feature]), xytext=(value[0+scene*feature], value[1+scene*feature]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

            plt.plot(value[3+scene*feature], value[4+scene*feature], "b.")
            circle1 = plt.Circle((value[3+scene*feature], value[4+scene*feature]), 0.105, fill=False)
            plt.gca().add_patch(circle1)
            x1=magnitude*np.cos(value[5+scene*feature])
            y1=magnitude*np.sin(value[5+scene*feature])
            plt.annotate("", xy=(x1+value[3+scene*feature], y1+value[4+scene*feature]), xytext=(value[3+scene*feature], value[4+scene*feature]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

            plt.plot(value[6+scene*feature], value[7+scene*feature], "r.")
            circle1 = plt.Circle((value[6+scene*feature], value[7+scene*feature]), 0.105, fill=False)
            plt.gca().add_patch(circle1)
            x1=magnitude*np.cos(value[8+scene*feature])
            y1=magnitude*np.sin(value[8+scene*feature])
            plt.annotate("", xy=(x1+value[6+scene*feature], y1+value[7+scene*feature]), xytext=(value[6+scene*feature], value[7+scene*feature]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

def elbow(i, X):
    plt.figure(i)
    sse = []

    for j in range(1,11):
        km = KMeans(n_clusters=j, algorithm='auto', random_state=42)
        km.fit(X)
        sse.append(km.inertia_)

    plt.plot(range(1,11), sse, marker='o')
    plt.xlabel('K')
    plt.ylabel('SSE')