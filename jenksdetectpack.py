import matplotlib.pyplot as plt
import numpy as np

# import jenkspy
# from jenks import jenks_breaks

def getJenksBreaks( dataList, numClass ):
    dataList.sort()
    mat1 = []
    for i in range(0,len(dataList)+1):
        temp = []
        for j in range(0,numClass+1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(0,len(dataList)+1):
        temp = []
        for j in range(0,numClass+1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1,numClass+1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2,len(dataList)+1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2,len(dataList)+1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1,l+1):
            i3 = l - m + 1
            val = float(dataList[i3-1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2,numClass+1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(dataList)
    kclass = []
    for i in range(0,numClass+1):
        kclass.append(0)
    kclass[numClass] = float(dataList[len(dataList) - 1])
    countNum = numClass
    while countNum >= 2:#print "rank = " + str(mat1[k][countNum])
        id = int((mat1[k][countNum]) - 2)
        kclass[countNum - 1] = dataList[id]
        k = int((mat1[k][countNum] - 1))
        countNum -= 1
    kclass[0] = dataList[0]
    return kclass

def goodness_of_variance_fit(array, classes):
    # get the break points
    class_num = classes
    array = np.array(array)
    # classes = jenkspy.jenks_breaks(array, nb_class=class_num)
    #classes = jenks_breaks(array, nb_class=classes)
    classes = getJenksBreaks(list(array), class_num)

    # classes = np.array(classes)

    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)

    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

    # sum of squared deviations of class means
    sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf, classified

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value <= breaks[i]:
            return i
    return len(breaks) - 1

color_map = {
    1: "blue",
    2: "green",
    3: "orange",
    4: "red",
    5: "black",
}

def jenks_until(data, plotting=False, plotting_all=False, cutoff=0.90):
    #input: mag means as data
    #output: data classified by jenks breaks
    #priors:
    #1 - data has noise and signal
    #2 - noise values are more similar to each other than signal
    #3 - noise corresponds to baseline values
    n_classes = 1
    assert len(data) > 2
    gvf = 0.0
    classified = None
    newcmp = None
    while (gvf < cutoff):
        n_classes += 1
        gvf, classified = goodness_of_variance_fit(data, n_classes)
        newcmp = None
        if plotting_all == True:
            newcmp = [color_map[a] for a in classified]
            fig = plt.figure()
            plt.title('GVF: ' + str(gvf) + " Class N: " + str(n_classes))
            plt.plot(data, color="red")
            plt.scatter([i for i,e in enumerate(data)], [e for i,e in enumerate(data)], c=newcmp, alpha=0.5)
            plt.show()    
    if plotting == True:
        print(classified)
        plt.figure()
        plt.title('Final - GVF: ' + str(gvf) + " Class N: " + str(n_classes))
        plt.plot(data, color="red")
        plt.scatter([i for i,e in enumerate(data)], [e for i,e in enumerate(data)], c=newcmp, alpha=0.5)
        plt.show()
    # quit()
    return gvf, classified
