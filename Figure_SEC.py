import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

def figure_timeseries(figure_title, raw_data, end):
    # raw_data.plot()
    INDEX = 7
    start = 0
    # end = raw_data.shape[0]
    end = end
    term = end - start
    plt.figure(figure_title)
    plt.xlim([start, end])
    # plt.ylabel("Physical Parameters")
    # plt.xlabel("Step")
    plt.figure(figsize=(15, 30))

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
    # plt.subplots(constrained_layout=True)
    # plt.subplot_tool()
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=1)
    plt.savefig(f"figures/{figure_title}.png")

def figure_cartesian_episode(figure_title, raw_data, episode_indexes):
    # plt.xlim([-0.5, 5.5])
    # plt.ylim([-0.5, 0.5])
    # plt.axis('scaled')
    
    maxsubfigure = 5
    print("maxsubfigure: ", maxsubfigure)
    indexes = []
    for _ in range(10):
        indexes.append([random.randint(1, np.array(episode_indexes).shape[0]) for _ in range(5)])
    print(np.array(indexes).shape)
    iii = 1
    for batch in indexes:
        plt.figure(f"heuristic_{iii}")
        plt.figure(figsize=(7, 8))
        ii = 1
        for episode in batch:
            magnitude = 0.21
            plt.subplot(5, 1, ii)
            plt.axis('scaled')
            plt.xlim([0, 5])
            plt.ylim([-0.5, 0.5])
            if ii == len(batch):
                plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            indexes = episode_indexes[episode]
            j = 1
            for index in indexes:
                if j == (len(indexes)):
                    plt.plot(raw_data["AEX"][index], raw_data["AEY"][index], "g.")
                    circle1 = plt.Circle((raw_data["AEX"][index], raw_data["AEY"][index]), 0.105, 
                                            fill=False, color='g')
                    plt.gca().add_patch(circle1)
                    x1=magnitude*np.cos(raw_data["AEZ"][index])
                    y1=magnitude*np.sin(raw_data["AEZ"][index])
                    plt.annotate("", xy=(x1+raw_data["AEX"][index], y1+raw_data["AEY"][index]), 
                                        xytext=(raw_data["AEX"][index], raw_data["AEY"][index]), 
                                        arrowprops={"facecolor": "green", 'edgecolor':'k', 'fill':'False',
                                                'shrink' : 0.2, 'alpha':1., 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                    plt.plot(raw_data["AAX"][index], raw_data["AAY"][index], "b.")
                    circle2 = plt.Circle((raw_data["AAX"][index], raw_data["AAY"][index]), 0.105, 
                                            fill=False, color='b')
                    plt.gca().add_patch(circle2)
                    x2=magnitude*np.cos(raw_data["AAZ"][index])
                    y2=magnitude*np.sin(raw_data["AAZ"][index])
                    plt.annotate("", xy=(x2+raw_data["AAX"][index], y2+raw_data["AAY"][index]), 
                                        xytext=(raw_data["AAX"][index], raw_data["AAY"][index]), 
                                        arrowprops={"facecolor": "blue", 'edgecolor':'k', 'fill':'False',
                                                'shrink' : 0.2, 'alpha':1., 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                    plt.plot(raw_data["ABX"][index], raw_data["ABY"][index], "r.")
                    circle3 = plt.Circle((raw_data["ABX"][index], raw_data["ABY"][index]), 0.105, 
                                            fill=False, color='r')
                    plt.gca().add_patch(circle3)
                    x3=magnitude*np.cos(raw_data["ABZ"][index])
                    y3=magnitude*np.sin(raw_data["ABZ"][index])
                    plt.annotate("", xy=(x3+raw_data["ABX"][index], y3+raw_data["ABY"][index]), 
                                        xytext=(raw_data["ABX"][index], raw_data["ABY"][index]), 
                                        arrowprops={"facecolor": "red", 'edgecolor':'k', 'fill':'False',
                                                'shrink' : 0.2, 'alpha':1., 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})
                    j += 1
                    plt.subplots_adjust(wspace=0, hspace=0.3)
                else:
                    plt.plot(raw_data["AEX"][index], raw_data["AEY"][index], "g.")
                    circle1 = plt.Circle((raw_data["AEX"][index], raw_data["AEY"][index]), 0.105, 
                                            fill=False, color='g', alpha=0.1)
                    plt.gca().add_patch(circle1)
                    x1=magnitude*np.cos(raw_data["AEZ"][index])
                    y1=magnitude*np.sin(raw_data["AEZ"][index])
                    plt.annotate("", xy=(x1+raw_data["AEX"][index], y1+raw_data["AEY"][index]), 
                                        xytext=(raw_data["AEX"][index], raw_data["AEY"][index]), 
                                        arrowprops={"facecolor": "green", 'edgecolor':'g', 
                                                'shrink' : 0.2, 'alpha':0.1, 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                    plt.plot(raw_data["AAX"][index], raw_data["AAY"][index], "b.")
                    circle2 = plt.Circle((raw_data["AAX"][index], raw_data["AAY"][index]), 0.105, 
                                            fill=False, color='b', alpha=0.1)
                    plt.gca().add_patch(circle2)
                    x2=magnitude*np.cos(raw_data["AAZ"][index])
                    y2=magnitude*np.sin(raw_data["AAZ"][index])
                    plt.annotate("", xy=(x2+raw_data["AAX"][index], y2+raw_data["AAY"][index]), 
                                        xytext=(raw_data["AAX"][index], raw_data["AAY"][index]), 
                                        arrowprops={"facecolor": "blue", 'edgecolor':'b', 
                                                'shrink' : 0.2, 'alpha':0.1, 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                    plt.plot(raw_data["ABX"][index], raw_data["ABY"][index], "r.")
                    circle3 = plt.Circle((raw_data["ABX"][index], raw_data["ABY"][index]), 0.105, 
                                            fill=False, color='r', alpha=0.1)
                    plt.gca().add_patch(circle3)
                    x3=magnitude*np.cos(raw_data["ABZ"][index])
                    y3=magnitude*np.sin(raw_data["ABZ"][index])
                    plt.annotate("", xy=(x3+raw_data["ABX"][index], y3+raw_data["ABY"][index]), 
                                        xytext=(raw_data["ABX"][index], raw_data["ABY"][index]), 
                                        arrowprops={"facecolor": "red", 'edgecolor':'r', 
                                                'shrink' : 0.2, 'alpha':0.1, 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})
                    j += 1
                    plt.subplots_adjust(wspace=0, hspace=0.3)
            ii += 1
            plt.subplots_adjust(wspace=0, hspace=0.3)
        plt.subplots_adjust(wspace=0, hspace=0.3)
        plt.savefig(f"figures/heuristic_{iii}.png")
        iii += 1
    plt.subplots_adjust(wspace=0, hspace=0.3)

def figure_clustering_algorithm(figure_title, dic):
    plt.figure(figure_title)
    plt.figure(figsize=(7, 8))
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
            # print(key, len(dic.keys()))
            if scene == (range_-1):
                if key == (len(dic.keys())-1):
                    plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.axis('scaled')
                plt.xlim([0, 5])
                # plt.ylim([-0.3, 0.3])
                plt.ylim([-0.5, 0.5])
                plt.plot(value[0+scene*feature], value[1+scene*feature], "g.")
                circle1 = plt.Circle((value[0+scene*feature], value[1+scene*feature]), 0.105, 
                                        fill=False, color='g')
                plt.gca().add_patch(circle1)
                x1=magnitude*np.cos(value[2+scene*feature])
                y1=magnitude*np.sin(value[2+scene*feature])
                plt.annotate("", xy=(x1+value[0+scene*feature], y1+value[1+scene*feature]), 
                                    xytext=(value[0+scene*feature], value[1+scene*feature]), 
                                    arrowprops={"facecolor": "green", 'edgecolor':'k', 'fill':'False',
                                                'shrink' : 0.2, 'alpha':1., 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                plt.plot(value[3+scene*feature], value[4+scene*feature], "b.")
                circle1 = plt.Circle((value[3+scene*feature], value[4+scene*feature]), 0.105, 
                                        fill=False, color='b')
                plt.gca().add_patch(circle1)
                x1=magnitude*np.cos(value[5+scene*feature])
                y1=magnitude*np.sin(value[5+scene*feature])
                plt.annotate("", xy=(x1+value[3+scene*feature], y1+value[4+scene*feature]), 
                                    xytext=(value[3+scene*feature], value[4+scene*feature]), 
                                    arrowprops={"facecolor": "blue", 'edgecolor':'k', 'fill':'False',
                                                'shrink' : 0.2, 'alpha':1., 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                plt.plot(value[6+scene*feature], value[7+scene*feature], "r.")
                circle1 = plt.Circle((value[6+scene*feature], value[7+scene*feature]), 0.105, 
                                        fill=False, color='r')
                plt.gca().add_patch(circle1)
                x1=magnitude*np.cos(value[8+scene*feature])
                y1=magnitude*np.sin(value[8+scene*feature])
                plt.annotate("", xy=(x1+value[6+scene*feature], y1+value[7+scene*feature]), 
                                    xytext=(value[6+scene*feature], value[7+scene*feature]), 
                                    arrowprops={"facecolor": "red", 'edgecolor':'k', 'fill':'False',
                                                'shrink' : 0.2, 'alpha':1., 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})
            else:
                if key == (len(dic.keys())-1):
                    plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.axis('scaled')
                plt.xlim([0, 5])
                # plt.ylim([-0.3, 0.3])
                plt.ylim([-0.5, 0.5])
                plt.plot(value[0+scene*feature], value[1+scene*feature], "g.")
                circle1 = plt.Circle((value[0+scene*feature], value[1+scene*feature]), 0.105, 
                                        fill=False, color='g', alpha=0.1)
                plt.gca().add_patch(circle1)
                x1=magnitude*np.cos(value[2+scene*feature])
                y1=magnitude*np.sin(value[2+scene*feature])
                plt.annotate("", xy=(x1+value[0+scene*feature], y1+value[1+scene*feature]), 
                                    xytext=(value[0+scene*feature], value[1+scene*feature]), 
                                    arrowprops={"facecolor": "green", 'edgecolor':'g', 
                                                'shrink' : 0.2, 'alpha':0.1, 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                plt.plot(value[3+scene*feature], value[4+scene*feature], "b.")
                circle1 = plt.Circle((value[3+scene*feature], value[4+scene*feature]), 0.105, 
                                        fill=False, color='b', alpha=0.1)
                plt.gca().add_patch(circle1)
                x1=magnitude*np.cos(value[5+scene*feature])
                y1=magnitude*np.sin(value[5+scene*feature])
                plt.annotate("", xy=(x1+value[3+scene*feature], y1+value[4+scene*feature]), 
                                    xytext=(value[3+scene*feature], value[4+scene*feature]), 
                                    arrowprops={"facecolor": "blue", 'edgecolor':'b', 
                                                'shrink' : 0.2, 'alpha':0.1, 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})

                plt.plot(value[6+scene*feature], value[7+scene*feature], "r.")
                circle1 = plt.Circle((value[6+scene*feature], value[7+scene*feature]), 0.105, 
                                        fill=False, color='r', alpha=0.1)
                plt.gca().add_patch(circle1)
                x1=magnitude*np.cos(value[8+scene*feature])
                y1=magnitude*np.sin(value[8+scene*feature])
                plt.annotate("", xy=(x1+value[6+scene*feature], y1+value[7+scene*feature]), 
                                    xytext=(value[6+scene*feature], value[7+scene*feature]), 
                                    arrowprops={"facecolor": "red", 'edgecolor':'r', 
                                                'shrink' : 0.2, 'alpha':0.1, 
                                                'headwidth':7, 'headlength':5, 
                                                'width':3})
    plt.subplots_adjust(wspace=0, hspace=0.3)
    # plt.subplots_adjust(wspace=0, hspace=0.0)
    plt.savefig(f"figures/{figure_title}.png")

def elbow(figure_title, X):
    plt.figure(figure_title)
    sse = []

    for j in range(1,21):
        km = KMeans(n_clusters=j, algorithm='auto', random_state=50)
        km.fit(X)
        sse.append(km.inertia_)

    plt.grid(True)
    plt.xticks([i for i in range(1, 21)])
    plt.plot(range(1,21), sse, marker='o')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.savefig(f"figures/{figure_title}.png")