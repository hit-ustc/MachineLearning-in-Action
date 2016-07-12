__author__ = 'CC'


from numpy import *
from LogRegression import *
import time

def loadData():
    train_x = []
    train_y = []
    fileIn = open('I:\\Code\\Python\\LR\\trainSet5.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(",")
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]),float(lineArr[3]), float(lineArr[4]), float(lineArr[5]), float(lineArr[6])
        , float(lineArr[7]), float(lineArr[8]), float(lineArr[9]), float(lineArr[10]), float(lineArr[11]), float(lineArr[12]), float(lineArr[13]), float(lineArr[14])
        , float(lineArr[15]), float(lineArr[16]), float(lineArr[17]), float(lineArr[18]), float(lineArr[19]), float(lineArr[20]), float(lineArr[21]), float(lineArr[22])
        , float(lineArr[23]), float(lineArr[24]), float(lineArr[25]), float(lineArr[26]), float(lineArr[27]), float(lineArr[28]), float(lineArr[29]), float(lineArr[30])
        , float(lineArr[31]), float(lineArr[32]), float(lineArr[33]), float(lineArr[34]), float(lineArr[35])])
        train_y.append(float(lineArr[36]))
    return mat(train_x), mat(train_y).transpose()

def loadTestData():
    test_x = []
    test_y = []
    fileIn = open('I:\\Code\\Python\\LR\\testSet5.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(",")
        test_x.append([1.0, float(lineArr[0]), float(lineArr[1]), float(lineArr[2]),float(lineArr[3]), float(lineArr[4]), float(lineArr[5]), float(lineArr[6])
        , float(lineArr[7]), float(lineArr[8]), float(lineArr[9]), float(lineArr[10]), float(lineArr[11]), float(lineArr[12]), float(lineArr[13]), float(lineArr[14])
        , float(lineArr[15]), float(lineArr[16]), float(lineArr[17]), float(lineArr[18]), float(lineArr[19]), float(lineArr[20]), float(lineArr[21]), float(lineArr[22])
        , float(lineArr[23]), float(lineArr[24]), float(lineArr[25]), float(lineArr[26]), float(lineArr[27]), float(lineArr[28]), float(lineArr[29]), float(lineArr[30])
        , float(lineArr[31]), float(lineArr[32]), float(lineArr[33]), float(lineArr[34]), float(lineArr[35])])
        test_y.append(float(lineArr[36]))
    return mat(test_x), mat(test_y).transpose()


## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x, test_y = loadTestData()

## step 2: training...
print "step 2: training..."
opts = {'alpha': 0.01, 'maxIter':500, 'optimizeType': 'gradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)