import numpy as np

print("tlqkf")
# for row in obj.rows:
#     for apm in ["AM", "PM"]:
#         for h in range(1, 12):
#             for m in range(1, 61):
#                 for s in range(1, 61):
#                     globals()["obu_{}_{}_{}_{}".format(apm, h, m, s)] = 
#             print(h)
# obu_1_2_3_4 = 1


# apm = 1
# h = 2
# m = 3
# s = 4
# globals()["obu_{}_{}_{}_{}".format(apm, h, m, s)] = 2
# if "obu_{}_{}_{}_{}".format(apm, h, m, s) in locals():
#     print("TRUE")
#     print("obu_{}_{}_{}_{}".format(apm, h, m, s))
# else:
#     print("FALSE")


# apm = obj["obu_msg_sent_time"][row].split(' ')
#         hms = apm[0].split(':')
#         apm = apm[1]
#         h = hms[0]
#         m = hms[1]
#         s = hms[2]
#         if "obu_{}_{}_{}_{}".format(apm, h, m, s) in globals():
#             pass
#         else:
#             globals()["obu_{}_{}_{}_{}".format(apm, h, m, s)] = 
#         globals()["obu_{}_{}_{}_{}".format(apm, h, m, s)] = 
#         for apm in ["AM", "PM"]:
            
#             for h in range(1, 12):
#                 for m in range(1, 61):
#                     for s in range(1, 61):
#                         globals()["obu_{}_{}_{}_{}".format(apm, h, m, s)] = 
#     for row in obu.rows:
        
#         shared_data_1(apm)
#         test = [test1[1]] + test2
#     return test

# def shared_data_1(apm):
    
#     if apm[1] == "AM":
#         # del obu_PM[row]
#         shared_data_2(hms)
#     elif apm[1] == "PM":
#         # del obu_AM[row]
#         shared_data_2(hms)
#     else:
#         "tlqkf"

# def shared_data_2(hms):
#     for h in range(1,13):
#         print(h)
#         h = str(h)
        
    # if hms[0] == "1":
    #     pass
    # elif hms[0] == "2":
    #     pass
    # elif hms[0] == "3":
    #     pass
    # elif hms[0] == "4":
    #     pass
    # elif hms[0] == "5":
    #     pass
    # elif hms[0] == "6":
    #     pass
    # elif hms[0] == "7":
    #     pass
    # elif hms[0] == "8":
    #     pass
    # elif hms[0] == "9":
    #     pass
    # elif hms[0] == "10":
    #     pass
    # elif hms[0] == "11":
    #     pass
    # elif hms[0] == "12":
    #     pass
    # else:
    #     "tlqkf"


# bitmap = [[[[[False] * 60] * 60] * 12] * 2]
# bitmap = np.array(bitmap[0])
# print(bitmap.shape)
# print(bitmap[0][0][0])
# bitmap[0][11][22][12] = True
# bitmap[1][1][2][32] = True
# args = np.where(bitmap == True)
# args = np.array(args)
# args = args.T

# print(args.shape)
# print(args[1][3])
# print(args[0])
# print(args)

# bitmap1 = np.array([[[[[False] * 60] * 60] * 12] * 2])
# bitmap2 = np.array([[[[[True] * 60] * 60] * 12] * 2])
# bitmap1 = np.array([[[False] * 2] * 2])[0]
# bitmap2 = np.array([[[False] * 2] * 2])[0]
# bitmap1 = list(bitmap1)
# bitmap2 = list(bitmap2)
# bitmap2[0][0] = True
# bitmap2[1][1] = True
# print(bitmap1, bitmap2)
# # bitmap = bitmap1.all() and bitmap2.all()
# bitmap = bitmap1 and bitmap2
# bitmap = np.array(bitmap)
# print(bitmap)

# import pandas as pd
# obu = pd.read_csv("PVD_200623_obu_state.csv")

# rsu_g1 = pd.read_csv("RSA_200623_rsu_accidentstate.csv")
# print(rsu_g1.head(100))
# rsu_g1 = rsu_g1.sort_values(by=["msg_received_time"], ascending=[False])
# print(rsu_g1.head(100))
# rsu_g2 = pd.read_csv("SPaT_200623_rsu_signalstate.csv")
# print(obu.shape)
# obu = obu.drop(2)
# print(obu.shape)

# l = [1, 2]
# ll = np.array([[1, 2], [2, 3]])
# print(type(l), type(ll))
# if l in ll.tolist():
#     print(l, ll)
#     print("tlqkf")

bitmap1 = [[[[[False] * 2] * 2] * 2] * 2]
bitmap1 = np.array(bitmap1[0])
tlqkf1 = [[[[[True] * 2] * 2] * 2] * 2]
tlqkf1 = np.array(tlqkf1[0])
temp = np.logical_and(bitmap1, tlqkf1)
print(temp)
# tlqkf = np.array(temp)
# tlqkf = np.where(tlqkf and True)
# tlqkf = np.array(tlqkf)
# tlqkf = tlqkf.T
# print(len(tlqkf))
# print(len(tlqkf), tlqkf[0])
