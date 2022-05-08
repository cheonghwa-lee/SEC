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