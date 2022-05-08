import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

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

def figure_timeseries_1000(i, raw_data, starts, ends):
    # raw_data.plot()
    INDEX = len(starts)
    upper = [0.21] * 35000
    lower = [1.00] * 35000
    # print(len(upper), upper)
    plt.figure(i)
    
    for idx in range(len(starts)):
        start = starts[idx] # 0
        end = ends[idx] # 5000
        term = end - start
        
        plt.subplot(INDEX, 1, idx+1)
        plt.xlim([start, end])
        plt.ylim([0, 2.0])
        plt.xlabel("Time Step")
        plt.ylabel("Distance")
        plt.grid(True)
        # print(start, end)
        # plt.plot(upper[start:end], "k--")
        # plt.plot(lower[start:end], "k--")
        plt.plot(raw_data["READ"][start:end], "b")
        plt.plot(raw_data["REBD"][start:end], "r")

def figure_timeseries_episodic_start_end(i, datadata, start, end):
    INDEX = 7
    plt.figure(i)
    for rd in datadata[start:end]:
        plt.subplot(INDEX, 1, 1)
        plt.ylabel("x")
        plt.grid(True)
        plt.ylim([0, 5])
        plt.plot(rd["step"][:], rd["AEX"][:], "g")
        plt.plot(rd["step"][:], rd["AAX"][:], "b")
        plt.plot(rd["step"][:], rd["ABX"][:], "r")

        plt.subplot(INDEX, 1, 2)
        plt.ylabel("y")
        plt.grid(True)
        plt.plot(rd["step"][:], rd["AEY"][:], "g")
        plt.plot(rd["step"][:], rd["AAY"][:], "b")
        plt.plot(rd["step"][:], rd["ABY"][:], "r")

        plt.subplot(INDEX, 1, 3)
        plt.ylabel("$\phi$")
        plt.grid(True)
        plt.plot(rd["step"][:], np.degrees(rd["AEZ"][:]), "g")
        plt.plot(rd["step"][:], np.degrees(rd["AAZ"][:]), "b")
        plt.plot(rd["step"][:], np.degrees(rd["ABZ"][:]), "r")

        plt.subplot(INDEX, 1, 4)
        plt.ylabel("$\Delta$x")
        plt.grid(True)
        plt.plot(rd["step"][:], rd["REAX"][:], "b")
        plt.plot(rd["step"][:], rd["REBX"][:], "r")
        # plt.plot([0.21]*term, "k--")
        # plt.plot([-0.21]*term, "k--")

        plt.subplot(INDEX, 1, 5)
        plt.ylabel("$\Delta$y")
        plt.grid(True)
        plt.plot(rd["step"][:], rd["REAY"][:], "b")
        plt.plot(rd["step"][:], rd["REBY"][:], "r")
        # plt.plot([0.21]*term, "k--")
        # plt.plot([-0.21]*term, "k--")

        plt.subplot(INDEX, 1, 6)
        plt.ylabel("$\Delta\phi$")
        plt.grid(True)
        plt.plot(rd["step"][:], np.degrees(rd["REAZ"][:]), "b")
        plt.plot(rd["step"][:], np.degrees(rd["REBZ"][:]), "r")

        plt.subplot(INDEX, 1, 7)
        plt.ylabel("d")
        plt.grid(True)
        # plt.plot([0.21]*term, "g--")
        # plt.plot([1.00]*term, "g--")
        plt.plot(rd["step"][:], rd["READ"][:], "b")
        plt.plot(rd["step"][:], rd["REBD"][:], "r")

def figure_cartesian(i, datadata):
    plt.figure(i)
    plt.xlim([0, 5])
    plt.ylim([-0.3, 0.3])
    # plt.xlim([-0.5, 5.5])
    # plt.ylim([-0.5, 0.5])
    plt.axis('scaled')
    ii = 1
    maxfigure = 10
    for rd in datadata[0:maxfigure]:
        magnitude = 0.21
        plt.subplot(maxfigure, 1, ii)
        plt.plot(rd["AEX"][:], rd["AEY"][:], "g.")
        x1=magnitude*np.cos(rd["AEZ"][:])
        y1=magnitude*np.sin(rd["AEZ"][:])
        # plt.annotate("", xy=(x1+rd["AEX"][:], y1+rd["AEY"][:]), xytext=(rd["AEX"][:], rd["AEY"][:]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(rd["AAX"][:], rd["AAY"][:], "b.")
        x2=magnitude*np.cos(rd["AAZ"][:])
        y2=magnitude*np.sin(rd["AAZ"][:])
        # plt.annotate("", xy=(x2+rd["AAX"][:], y2+rd["AAY"][:]), xytext=(rd["AAX"][:], rd["AAY"][:]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(rd["ABX"][:], rd["ABY"][:], "r.")
        x3=magnitude*np.cos(rd["ABZ"][:])
        y3=magnitude*np.sin(rd["ABZ"][:])
        # plt.annotate("", xy=(x3+rd["ABX"][:], y3+rd["ABY"][:]), xytext=(rd["ABX"][:], rd["ABY"][:]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})
        ii += 1

def figure_cartesian_single(i, conditions):
    plt.figure(i)
    # plt.xlim([0, 5])
    # plt.ylim([-0.3, 0.3])
    # plt.axis('scaled')
    magnitude = 0.21
    ii = 1
    for condition in conditions:
        feature = 9
        range_ = int(len(condition)/feature)
        # print("range: ", range_)
        for scene in range(range_):
            plt.subplot(len(conditions), 1, ii)
            plt.axis('scaled')
            plt.xlim([-1, 6])
            plt.ylim([-0.75, 0.75])
            # print(condition[0], condition[1])
            plt.plot(condition[0+scene*feature], condition[1+scene*feature], "g.")
            # print(0+scene, 1+scene)
            circle1 = plt.Circle((condition[0+scene*feature], condition[1+scene*feature]), 0.105, fill=False)
            plt.gca().add_patch(circle1)
            x1=magnitude*np.cos(condition[2+scene*feature])
            y1=magnitude*np.sin(condition[2+scene*feature])
            plt.annotate("", xy=(x1+condition[0+scene*feature], y1+condition[1+scene*feature]), xytext=(condition[0+scene*feature], condition[1+scene*feature]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

            plt.plot(condition[3+scene*feature], condition[4+scene*feature], "b.")
            circle2 = plt.Circle((condition[3+scene*feature], condition[4+scene*feature]), 0.105, fill=False)
            plt.gca().add_patch(circle2)
            x2=magnitude*np.cos(condition[5+scene*feature])
            y2=magnitude*np.sin(condition[5+scene*feature])
            plt.annotate("", xy=(x2+condition[3+scene*feature], y2+condition[4+scene*feature]), xytext=(condition[3+scene*feature], condition[4+scene*feature]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

            plt.plot(condition[6+scene*feature], condition[7+scene*feature], "r.")
            circle3 = plt.Circle((condition[6+scene*feature], condition[7+scene*feature]), 0.105, fill=False)
            plt.gca().add_patch(circle3)
            x3=magnitude*np.cos(condition[8+scene*feature])
            y3=magnitude*np.sin(condition[8+scene*feature])
            plt.annotate("", xy=(x3+condition[6+scene*feature], y3+condition[7+scene*feature]), xytext=(condition[6+scene*feature], condition[7+scene*feature]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})
        ii += 1

def figure_cartesian_start_end(i, datadata, start_f, end_f):
    plt.figure(i)
    # plt.figure(figsize=(12, 3))
    plt.xlim([0, 5])
    plt.ylim([-0.3, 0.3])
    # plt.axis('scaled')
    ii = 1
    start_f = 0
    end_f = 10
    maxfigure = end_f - start_f
    # conditions = []
    for rd in datadata[start_f:end_f]:
        init_c = rd.index[rd["step"] == 1]
        finish_c = rd.index[rd["step"] == len(rd)-1] # 
        magnitude = 0.21
        plt.subplot(maxfigure, 1, ii)
        # plt.axis('scaled')
        plt.xlim([0, 5])
        plt.ylim([-0.3, 0.3])
        plt.plot(rd["AEX"][init_c], rd["AEY"][init_c], "g.")
        circle1 = plt.Circle((rd["AEX"][init_c], rd["AEY"][init_c]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(rd["AEZ"][init_c])
        y1=magnitude*np.sin(rd["AEZ"][init_c])
        plt.annotate("", xy=(x1+rd["AEX"][init_c], y1+rd["AEY"][init_c]), xytext=(rd["AEX"][init_c], rd["AEY"][init_c]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(rd["AAX"][init_c], rd["AAY"][init_c], "b.")
        circle2 = plt.Circle((rd["AAX"][init_c], rd["AAY"][init_c]), 0.105, fill=False)
        plt.gca().add_patch(circle2)
        x2=magnitude*np.cos(rd["AAZ"][init_c])
        y2=magnitude*np.sin(rd["AAZ"][init_c])
        plt.annotate("", xy=(x2+rd["AAX"][init_c], y2+rd["AAY"][init_c]), xytext=(rd["AAX"][init_c], rd["AAY"][init_c]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(rd["ABX"][init_c], rd["ABY"][init_c], "r.")
        circle3 = plt.Circle((rd["ABX"][init_c], rd["ABY"][init_c]), 0.105, fill=False)
        plt.gca().add_patch(circle3)
        x3=magnitude*np.cos(rd["ABZ"][init_c])
        y3=magnitude*np.sin(rd["ABZ"][init_c])
        plt.annotate("", xy=(x3+rd["ABX"][init_c], y3+rd["ABY"][init_c]), xytext=(rd["ABX"][init_c], rd["ABY"][init_c]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        # plt.subplot(maxfigure, 2, ii)
        plt.axis('scaled')
        plt.xlim([0, 5])
        plt.ylim([-0.3, 0.3])
        plt.plot(rd["AEX"][finish_c], rd["AEY"][finish_c], "g.")
        circle4 = plt.Circle((rd["AEX"][finish_c], rd["AEY"][finish_c]), 0.105, fill=False)
        plt.gca().add_patch(circle4)
        x1=magnitude*np.cos(rd["AEZ"][finish_c])
        y1=magnitude*np.sin(rd["AEZ"][finish_c])
        plt.annotate("", xy=(x1+rd["AEX"][finish_c], y1+rd["AEY"][finish_c]), xytext=(rd["AEX"][finish_c], rd["AEY"][finish_c]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(rd["AAX"][finish_c], rd["AAY"][finish_c], "b.")
        circle5 = plt.Circle((rd["AAX"][finish_c], rd["AAY"][finish_c]), 0.105, fill=False)
        plt.gca().add_patch(circle5)
        x2=magnitude*np.cos(rd["AAZ"][finish_c])
        y2=magnitude*np.sin(rd["AAZ"][finish_c])
        plt.annotate("", xy=(x2+rd["AAX"][finish_c], y2+rd["AAY"][finish_c]), xytext=(rd["AAX"][finish_c], rd["AAY"][finish_c]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(rd["ABX"][finish_c], rd["ABY"][finish_c], "r.")
        circle6 = plt.Circle((rd["ABX"][finish_c], rd["ABY"][finish_c]), 0.105, fill=False)
        plt.gca().add_patch(circle6)
        x3=magnitude*np.cos(rd["ABZ"][finish_c])
        y3=magnitude*np.sin(rd["ABZ"][finish_c])
        plt.annotate("", xy=(x3+rd["ABX"][finish_c], y3+rd["ABY"][finish_c]), xytext=(rd["ABX"][finish_c], rd["ABY"][finish_c]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})
        ii += 1

        # conditions = [rd["AEX"][finish_c], rd["AEY"][finish_c], rd["AEZ"][finish_c], rd["AAX"][finish_c], rd["AAY"][finish_c], rd["AAZ"][finish_c], rd["ABX"][finish_c], rd["ABY"][finish_c], rd["ABZ"][finish_c]]

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

def figure_clustering_algorithm_series(i, dic):
    plt.figure(i)
    # plt.xlim([-1, 6])
    # plt.ylim([-0.3, 0.3])
    for key in dic.keys():
        value = dic[key]
        # finish_c = rd.index[rd["step"] == len(rd)-1] # 
        magnitude = 0.21
        plt.subplot(len(dic), 1, key +1)
        plt.axis('scaled')
        plt.xlim([-1, 6])
        # plt.ylim([-0.3, 0.3])
        plt.ylim([-0.75, 0.75])
        plt.plot(value[0], value[1], "g.")
        circle1 = plt.Circle((value[0], value[1]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(value[2])
        y1=magnitude*np.sin(value[2])
        plt.annotate("", xy=(x1+value[0], y1+value[1]), xytext=(value[0], value[1]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(value[3], value[4], "b.")
        circle1 = plt.Circle((value[3], value[4]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(value[5])
        y1=magnitude*np.sin(value[5])
        plt.annotate("", xy=(x1+value[3], y1+value[4]), xytext=(value[3], value[4]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(value[6], value[7], "r.")
        circle1 = plt.Circle((value[6], value[7]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(value[8])
        y1=magnitude*np.sin(value[8])
        plt.annotate("", xy=(x1+value[6], y1+value[7]), xytext=(value[6], value[7]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

def figure_cartesian_random(i, conditions_abs):
    indexes = [random.randint(0,len(conditions_abs)-1) for _ in range(5)]
    # print(indexes)
    plt.figure(i)
    plt.xlabel("x")
    plt.ylabel("y")
    iii = 1
    for index in indexes:
        magnitude = 0.21
        plt.subplot(len(indexes), 1, iii)
        plt.axis('scaled')
        plt.xlim([-1, 6])
        plt.ylim([-0.3, 0.3])
        plt.plot(conditions_abs[index][0], conditions_abs[index][1], "g.")
        circle1 = plt.Circle((conditions_abs[index][0], conditions_abs[index][1]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(conditions_abs[index][2])
        y1=magnitude*np.sin(conditions_abs[index][2])
        plt.annotate("", xy=(x1+conditions_abs[index][0], y1+conditions_abs[index][1]), xytext=(conditions_abs[index][0], conditions_abs[index][1]), arrowprops={"facecolor": "green", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(conditions_abs[index][3], conditions_abs[index][4], "b.")
        circle1 = plt.Circle((conditions_abs[index][3], conditions_abs[index][4]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(conditions_abs[index][5])
        y1=magnitude*np.sin(conditions_abs[index][5])
        plt.annotate("", xy=(x1+conditions_abs[index][3], y1+conditions_abs[index][4]), xytext=(conditions_abs[index][3], conditions_abs[index][4]), arrowprops={"facecolor": "blue", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})

        plt.plot(conditions_abs[index][6], conditions_abs[index][7], "r.")
        circle1 = plt.Circle((conditions_abs[index][6], conditions_abs[index][7]), 0.105, fill=False)
        plt.gca().add_patch(circle1)
        x1=magnitude*np.cos(conditions_abs[index][8])
        y1=magnitude*np.sin(conditions_abs[index][8])
        plt.annotate("", xy=(x1+conditions_abs[index][6], y1+conditions_abs[index][7]), xytext=(conditions_abs[index][6], conditions_abs[index][7]), arrowprops={"facecolor": "red", 'edgecolor':'k', 'shrink' : 0.2, 'alpha':0.5})
        iii += 1

def figure_cartesian_3d(i, conditions_abs):
    fig = plt.figure(i)
    figure = fig.add_subplot(111, projection='3d')
    for index in range(len(conditions_abs)):
        figure.scatter(conditions_abs[index][0], conditions_abs[index][1], conditions_abs[index][2], color = 'g', alpha = 0.5)
        figure.scatter(conditions_abs[index][3], conditions_abs[index][4], conditions_abs[index][5], color = 'b', alpha = 0.5)
        figure.scatter(conditions_abs[index][6], conditions_abs[index][7], conditions_abs[index][8], color = 'r', alpha = 0.5)

def figure_cartesian_2d(i, conditions_abs):
    fig = plt.figure(i)
    figure = fig.add_subplot(111, projection='3d')
    for index in range(len(conditions_abs)):
        figure.scatter(conditions_abs[index][0], conditions_abs[index][1], conditions_abs[index][2], color = 'b', alpha = 0.5)
        figure.scatter(conditions_abs[index][3], conditions_abs[index][4], conditions_abs[index][5], color = 'r', alpha = 0.5)

    plt.figure(i+200)
    for index in range(len(conditions_abs)):
        plt.plot(conditions_abs[index][0], conditions_abs[index][1], "g.")
        plt.plot(conditions_abs[index][3], conditions_abs[index][4], "b.")
        plt.plot(conditions_abs[index][6], conditions_abs[index][7], "r.")