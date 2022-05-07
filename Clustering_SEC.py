import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
