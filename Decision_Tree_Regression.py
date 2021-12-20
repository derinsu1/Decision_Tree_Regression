import numpy as np
import csv
import matplotlib.pyplot as plt
import os.path
from scipy import signal


def readData(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x = []
        y = []
        for index, i in enumerate(csv_reader):
            if index == 0:
                for j in i:
                    x.append(float(j))
            else:
                for j in i:
                    y.append(float(j))
        return x, y


def createDataset():
    datasetCoordinates = []  # 0: Sinusoidal, 1: second degree polynomial, 2: linear line, 3: upside down triangle
    testSetCoordinates = []  # 0: Sinusoidal, 1: second degree polynomial, 2: linear line, 3: upside down triangle
    # Generating noisy sinusoidal signal
    x = np.linspace(0, 2 * np.pi, num=201)
    xT = np.linspace(0.1, 2 * np.pi - 0.1, num=39)
    r = np.random.normal(scale=20, size=x.size)    # Random sample with Gaussian distribution for dataset
    rT = np.random.normal(scale=20, size=xT.size)  # Random sample with Gaussian distribution for test set
    y = 50 * np.sin(x) + r + 40
    yT = 50 * np.sin(xT) + rT + 40
    np.savetxt("dataset0.csv", [x, y], delimiter=",")
    np.savetxt("dataset0Test.csv", [xT, yT], delimiter=",")
    datasetCoordinates.append([x, y])
    testSetCoordinates.append([xT, yT])
    # Generating noisy second degree polynomial signal
    x = np.linspace(0, 100, num=201)
    xT = np.linspace(2, 98, num=39)
    y = []
    yT = []
    r = np.random.normal(100, scale=5, size=x.size)    # Random sample with Gaussian distribution for dataset
    rT = np.random.normal(100, scale=5, size=xT.size)  # Random sample with Gaussian distribution for test set
    for i in range(len(x)):
        yy = -0.03 * x[i] * x[i] - 1 * x[i] + 650
        yy += yy * ((r[i] - 100) / 100)
        y.append(yy)
    for i in range(len(xT)):
        yyT = -0.03 * xT[i] * xT[i] - 1 * xT[i] + 650
        yyT += yyT * ((rT[i] - 100) / 100)
        yT.append(yyT)
    np.savetxt("dataset1.csv", [x, y], delimiter=",")
    np.savetxt("dataset1Test.csv", [xT, yT], delimiter=",")
    datasetCoordinates.append([x, y])
    testSetCoordinates.append([xT, yT])
    # Generating noisy linear line signal
    y = []
    yT = []
    r = np.random.normal(100, scale=2, size=x.size)    # Random sample with Gaussian distribution for dataset
    rT = np.random.normal(100, scale=2, size=xT.size)  # Random sample with Gaussian distribution for test set
    for i in range(len(x)):
        yy = (0.3 * x[i] + 50)
        yy += r[i] - 100
        y.append(yy)
    for i in range(len(xT)):
        yyT = (0.3 * xT[i] + 50)
        yyT += rT[i] - 100
        yT.append(yyT)
    np.savetxt("dataset2.csv", [x, y], delimiter=",")
    np.savetxt("dataset2Test.csv", [xT, yT], delimiter=",")
    datasetCoordinates.append([x, y])
    testSetCoordinates.append([xT, yT])
    # Generating noisy triangle wave signal
    x = np.linspace(0, 2, 201)
    xT = np.linspace(0.04, 1.96, 39)
    r = np.random.normal(0.1, scale=0.5, size=x.size)    # Random sample with Gaussian distribution for dataset
    rT = np.random.normal(0.1, scale=0.5, size=xT.size)  # Random sample with Gaussian distribution for test set
    y = signal.sawtooth(np.pi * x + 3 + r, 0.5)
    yT = signal.sawtooth(np.pi * xT + 3 + rT, 0.5)
    np.savetxt("dataset3.csv", np.array([x, list(y)]), delimiter=",")
    np.savetxt("dataset3Test.csv", np.array([xT, list(yT)]), delimiter=",")
    datasetCoordinates.append([x, list(y)])
    testSetCoordinates.append([xT, list(yT)])

    return datasetCoordinates, testSetCoordinates


def meanSquaredError(y, calculatedY):
    error = 0
    for index, value in enumerate(calculatedY):
        error += (value - y[index]) ** 2
    return error / len(calculatedY)


def decisionTreeRegression(x, y, depth, maxDepth, minSamples, splits):
    if depth == maxDepth or len(x) / 2 < minSamples:
        splits.append([x, y])
        return splits
    bestSplit = -1  # y = ?
    lowestError = meanSquaredError([sum(y) / len(y)] * len(y), y)
    for i in range(minSamples, len(x) - minSamples - 1):
        ySplit1 = y[:i]
        ySplit2 = y[i:]
        meanSplit1 = [sum(ySplit1) / len(ySplit1)] * len(ySplit1)
        meanSplit2 = [sum(ySplit2) / len(ySplit2)] * len(ySplit2)
        error = meanSquaredError(meanSplit1 + meanSplit2, y)
        if error < lowestError:
            lowestError = error
            bestSplit = i

    if bestSplit == -1:
        splits.append([x, y])
        return splits

    depth += 1
    splits = decisionTreeRegression(x[:bestSplit], y[:bestSplit], depth, maxDepth, minSamples, splits)
    splits = decisionTreeRegression(x[bestSplit:], y[bestSplit:], depth, maxDepth, minSamples, splits)
    return splits


def plotTreeRegression(data, test, maxDepth, minSamples):
    plt.figure(figsize=(12, 7))
    prevHighestX = data[0][0][0]    # Add offset to get continuous decision lines
    testDataPrediction = []
    for i in range(len(data)):
        if len(data[i][0]) == 0:
            continue
        xOffset = data[i][0][0] - prevHighestX
        x = [data[i][0][0] - xOffset, data[i][0][len(data[i][0]) - 1]]
        prevHighestX = data[i][0][len(data[i][0]) - 1]
        y = [sum(data[i][1]) / len(data[i][1]), sum(data[i][1]) / len(data[i][1])]
        for tx in test[0]:
            if x[0] <= tx < x[1]:
                testDataPrediction.append(y[0])
        # Plot data points
        plt.scatter(data[i][0], data[i][1], label="{:.3f}".format(y[0]), alpha=0.75, edgecolor='white')
        # Plot tree regions
        plt.plot(x, y, color='black', linewidth=2)

    # Find total MSE
    allDataY = []
    allDataPrediction = []
    for d in data:
        allDataY = allDataY + list(d[1])
        allDataPrediction += [sum(d[1]) / len(d[1])] * len(d[1])
    totalError = meanSquaredError(allDataY, allDataPrediction)

    # Plot test data
    plt.scatter(test[0], test[1], color='black', alpha=0.5, edgecolor='white', s=50)
    # Find test MSE
    testError = meanSquaredError(test[1], testDataPrediction)

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title("Decision Tree Regression  -  Max Depth=%i  -  Min Samples=%i  -  Mean Square Error=%f  -  Test MSE=%f" %
              (maxDepth, minSamples, totalError, testError))
    plt.show()
    return totalError, testError


def plotParameterComparison(trainErrors, testErrors, datasetName):
    maxDepth = [2, 3, 4, 5, 10]
    plt.figure(figsize=(12, 7))
    plt.xlabel("Maximum Depth")
    plt.ylabel("Mean Square Error")
    plt.title("Dataset: %s" % datasetName)
    plt.plot(maxDepth, trainErrors, marker='o', color='orange', label="Error on the training set")
    plt.plot(maxDepth, testErrors, marker='o', label="Error on the test set")
    plt.legend()
    plt.show()


def main():
    if os.path.isfile('dataset0.csv'):  # If you want to recreate the datasets, delete dataset0.csv file
        datasets = [readData('dataset0.csv'), readData('dataset1.csv'),
                    readData('dataset2.csv'), readData('dataset3.csv')]
        testSets = [readData('dataset0Test.csv'), readData('dataset1Test.csv'),
                    readData('dataset2Test.csv'), readData('dataset3Test.csv')]
    else:
        datasets, testSets = createDataset()

    datasetNames = ["Noisy Sinusoidal Signal", "Noisy Second Degree Polynomial",
                    "Noisy Linear Function", "Noisy Upside Down Triangle"]

    for i, d in enumerate(datasets):
        trainErrors = []
        testErrors = []
        DTR = decisionTreeRegression(d[0], d[1], 0, 2, 14, [])
        trainErr, testErr = plotTreeRegression(DTR, [testSets[i][0], testSets[i][1]], 2, 14)
        trainErrors.append(trainErr)
        testErrors.append(testErr)

        DTR = decisionTreeRegression(d[0], d[1], 0, 3, 10, [])
        trainErr, testErr = plotTreeRegression(DTR, [testSets[i][0], testSets[i][1]], 3, 10)
        trainErrors.append(trainErr)
        testErrors.append(testErr)

        DTR = decisionTreeRegression(d[0], d[1], 0, 4, 8, [])
        trainErr, testErr = plotTreeRegression(DTR, [testSets[i][0], testSets[i][1]], 4, 8)
        trainErrors.append(trainErr)
        testErrors.append(testErr)

        DTR = decisionTreeRegression(d[0], d[1], 0, 5, 5, [])
        trainErr, testErr = plotTreeRegression(DTR, [testSets[i][0], testSets[i][1]], 5, 5)
        trainErrors.append(trainErr)
        testErrors.append(testErr)

        DTR = decisionTreeRegression(d[0], d[1], 0, 10, 3, [])
        trainErr, testErr = plotTreeRegression(DTR, [testSets[i][0], testSets[i][1]], 10, 3)
        trainErrors.append(trainErr)
        testErrors.append(testErr)

        plotParameterComparison(trainErrors, testErrors, datasetNames[i])


if __name__ == "__main__":
    main()
