# 查看在我们铜片上的算法的表现


# 1 gamma 处理

# gamma
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
from scipy.interpolate import UnivariateSpline

IMAGE_NEED = 3


# 定义Gamma矫正的函数
def gamma_trans(img, gamma):
    """
    :param img: 输入的图像
    :param gamma:  矫正图像所使用的gamma系数
    :return:  返回一张矫正好的图像
    """
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)


def findSplit(b):
    u = b.mean()
    temp = []
    # c = b.copy()
    # c.sort()
    # c = np.unique(c)
    # #     print(c)
    c = np.linspace(b.min(), b.max(), num=20, endpoint=False)
    for i in c[1:]:
        x1 = b[b < i]
        y1 = b[b >= i]
        m1 = x1.__len__()
        m2 = y1.__len__()
        temp.append(m1 * (x1.mean() - u) ** 2 + m2 * (y1.mean() - u) ** 2)

    temp = np.array(temp)
    split = c[np.argmax(temp)]
    return split, temp


def getLocalWidth(b, Cols, plotSplit=True):
    # b: interval of x
    # x : mean of some rows of image
    # b = x[427:501]
    # b = x[649:742]
    split, temp = findSplit(b)

    # 只选择那些split point 以上的线段只有一段或者两段的统计宽度
    point = np.sum(np.diff(b >= split) == 1)
    if point > 2:
        return -1

    # splitInterval : 一段内大于阈值的部分
    splitInterval = (b > split).astype(np.int)
    splitInterval = np.concatenate([splitInterval, [0]])
    # x = [0,0,1,1,0,0,1,1,1,1,0]
    begin = 0
    end = 0
    beginMax = 0
    endMax = 0
    for i in range(len(splitInterval)):
        if splitInterval[i] == 0:

            # 判断是否更新最优间隔
            if end - begin >= endMax - beginMax:
                # 记录当前最优的间隔：
                beginMax = begin
                endMax = end
            # 重置begin,end的位置
            begin = i
            end = i
        else:
            end = i

    ironWidth = endMax - beginMax

    if plotSplit:
        print(Cols, ironWidth)
        plt.plot(temp)
        plt.show()
        plt.plot(b)
        plt.axhline(split)
        plt.show()

    return ironWidth


def searchEdge(ironSub, classifier, plot=False, bias=20):
    """

    :param ironSub: 一张传入进去的子图
    :param classifier:  进行边缘和铜片识别的分类器
    :param plot:  是否要将边缘识别的结果可视化出来
    :param bias:  bias ，找到bias后向两侧稍作偏移，避免倾斜造成的误判
    :return: 返回两个边缘点
    """
    edge = classifier.predict(ironSub.reshape(-1, 3)).reshape(ironSub[..., 0].shape).mean(0) > 0.5
    edgePoints = np.argwhere(np.diff(edge) == 1).ravel()
    try:
        maxSub = np.argmax(np.diff(edgePoints))
    except:
        assert 1 != 2
    edgeP = [edgePoints[maxSub], edgePoints[maxSub + 1]]
    edgeP = [edgeP[0] - bias, edgeP[1] + bias]
    if plot:
        figure = plt.figure(figsize=(15, 15))
        plt.imshow(ironSub)

        # 注意：实际上去取范围不会取的这么大
        plt.axvline(edgeP[0], color='r', linewidth=4)
        plt.axvline(edgeP[1], color='r', linewidth=4)
    # 考虑到倾斜，加入bias
    return edgeP


def showResult(classifier,img,imgC,plotNum,plotCounter,
               startRow=500, rowNum=100,
               d=4,
               argrelmaxOrder=10, numOfStopping=20, scale=1.3
               , plot=True, plotRowNum=40, coords=False, useMax=True,record = True
               ):
    """

    :param classifier: 边缘分类器
    :param startRow:  本次计数所使用的开始的位置
    :param rowNum:  本次计数使用的行数
    :param d:  进行差分的间隔
    :param argrelmaxOrder:  初始化寻找局部最大值的参数
    :param numOfStopping:  迭代次数
    :param scale:  放缩比例
    :param plot: 是否进行画图
    :param plotRowNum:  plot参数
    :return: 返回一系列的坐标
    """
    # 先找到边缘
    # edge = searchEdge(imgC[startRow:(startRow + rowNum), :], classifier, plot=True, bias=5)
    # print(edge)

    countCol = slice(0, -1)
    # countCol = slice(edge[0], edge[1])

    # 局部计数
    # cv2.imwrite("img_600_900.png",img[400:410,600:900])

    # x = img[startRow:(startRow + rowNum), countCol].mean(0)

    '''
    testing code here
    '''
    # lr = joblib.load("logisticForIronPad.model")
    # lr.predict(imgC[startRow:(startRow + rowNum), countCol].mean(0))
    # # plt.imshow(imgC[startRow:(startRow + 50), countCol])
    # predictAsPad = lr.predict(imgC[startRow:(startRow + rowNum), countCol].mean(0))
    # # plt.plot(predictAsPad*50)
    imgc = img[:, countCol].copy()
    # imgc[:,predictAsPad == 1] = 0
    x = imgc[startRow:(startRow + rowNum), countCol].mean(0)

    '''
    plt.imshow(imgc)
    x = imgc[startRow:(startRow + rowNum), countCol].mean(0)
    plt.plot(-x*500)

    '''

    '''
    testing code here
    '''

    # cv2.imwrite("img_" + str(startRow) + "_100_2000.png", img[startRow:(startRow + rowNum), countCol])

    diff = x[d:] - x[:-d]

    '''
    plt.imshow(imgc)
    plt.plot(-diff)
    plt.show()
    '''
    np.random.seed(10)
    diff = diff + np.random.random(diff.shape[0]) * 0.0001
    # plt.imshow( img[startRow:(startRow +40), countCol])
    # plt.plot(diff)

    currentCount = 0
    flag = 0
    xcoords = []
    stopWhile = 0

    while (True):
        stopWhile += 1
        xcoords = argrelmin(diff, order=argrelmaxOrder)[0]
        diffXcocrds = np.diff(xcoords)
        low = np.percentile(diffXcocrds, 15)
        high = np.percentile(diffXcocrds, 85)
        #         print(diffXcocrds)
        mean = (diffXcocrds[(diffXcocrds >= low) * (diffXcocrds <= high)]).mean()
        try:
            argrelmaxOrder = int(scale * mean / 2)
        except:
            print("get error")
            assert 1 != 2

        if currentCount == xcoords.__len__():
            flag += 1
        else:
            currentCount = xcoords.__len__()
        if flag >= numOfStopping or stopWhile > 30:
            break

    if coords:
        return xcoords
    else:

        '''
        查看最原始的分割结果对应的图片：
            counter = None
            fig = plt.figure(figsize=(20,3))
            plt.title("counter : " + str(counter))
            plt.imshow(imgC[startRow:(startRow + plotRowNum), countCol])
            plt.plot(diff*100)
            for xc in xcoords:
                plt.axvline(x=xc, color='r',linewidth = 4)
            print(plotNum)
            plt.show()


       '''

        interval = np.diff(xcoords)
        firstOutpoint = interval.mean() - interval.std() * 3
        while True:
            interval = np.diff(xcoords)
            correct2 = interval[interval < firstOutpoint]
            for i in correct2:
                locate = np.argwhere(interval == i).ravel()[0]
                # 去除不需要的间隔过小的点
                xcoords = np.delete(xcoords, locate + 1)
            if correct2.shape[0] == 0:
                break

        # 获得修正后的xcoords，消除了异常的小间隔
        # 再获取铜片宽度
        ironWidthList = []
        for j in range(xcoords.__len__() - 1):
            # for j in [17]:
            # b : 拿到了一个间隔中的所有的原始的像素点
            b = x[xcoords[j]: xcoords[j + 1]]

            '''
            找到b对应的那段图像是什么样的

            plt.imshow( imgC[startRow:(startRow+50),xcoords[j]: xcoords[j + 1]])
            plt.plot(b*50)

           '''
            ironWidthList.append(getLocalWidth(b, (xcoords[j], xcoords[j + 1]), plotSplit=False))

        ironWidthList = np.array(ironWidthList)
        ironWidthList = ironWidthList[ironWidthList != -1]
        ironWidthList.sort()
        ironWidth = np.median(ironWidthList)

        counter = xcoords.__len__()

        correct = interval[interval > interval.mean() + interval.std() * 2]

        # 利用铜片宽度，消除异常：
        if correct.shape[0] != 0:
            for index in range(len(correct)):
                # 定位:
                loc = np.argwhere(interval == correct[index]).ravel()[0]
                doubtInterval = x[xcoords[loc]:xcoords[loc + 1]]

                # plt.imshow(imgC[startRow :(startRow + 50) , xcoords[loc]:xcoords[loc + 1]])
                # plt.plot(doubtInterval)

                split, _ = findSplit(doubtInterval)

                # 获得大于split的部分：
                correct3 = round((doubtInterval >= split).sum() / ironWidth) - 1
                counter += correct3 if correct3 > 0 else 0
                # plt.plot(doubtInterval)
                # plt.imshow(imgC[startRow:(startRow+plotRowNum),xcoords[loc]:xcoords[loc + 1],:])
        # 考虑是否进行最终结果的作图
        if plot:
            # global plotNum, plotCounter

            fig = plt.figure(figsize=(20, 20))

            plt.title('startRow ' + str(startRow) +
                      ' rowNum ' + str(rowNum) +
                      ' d ' + str(d) +
                      ' scale ' + str(scale) +
                      ' argrelmaxOrder' + str(argrelmaxOrder) +
                      ' correct counter ' + str(counter) +
                      ' counter ' + str(xcoords.__len__())
                      )

            plt.plot(diff)
            for xc in xcoords:
                plt.axvline(x=xc, color='r')
            plt.show()

            fig = plt.figure(figsize=(20, 3))
            plt.title("counter : " + str(counter))
            plt.imshow(imgC[startRow:(startRow + plotRowNum), countCol])
            for xc in xcoords:
                plt.axvline(x=xc, color='r', linewidth=4)
            print(plotNum)
            plt.show()
        if record:
            tempResult = {"reference" : { "startRow":startRow , "xcoords":xcoords ,"xcoordsLen":xcoords.shape[0] } ,"counter":counter}
            return tempResult
        return counter


def derectionCorrect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(img, 15, 75, 75)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
    savePlot = False

    if savePlot:
        imgForSave = img.copy()
        for line in lines:
            line = line.ravel()
            rho = line[0]
            theta = line[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(imgForSave, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite("imgAfterCorrect.jpg", imgForSave)

    theta = lines.reshape(-1, 2)[:, 1]
    theta_high = np.percentile(theta, 85)
    theta_low = np.percentile(theta, 15)
    thetaMean = theta[(theta <= theta_high) * (theta >= theta_low)].mean()

    rows, cols, _ = img.shape
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2),    - thetaMean  , 1.1)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), thetaMean * 180 / 3.1415926 - 90, 1.1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst



def getIntervalOfIron(cut,img_diff_abs,maxInterval = 100,bandWidth = 10):
    getCoords = np.argwhere(img_diff_abs > cut).ravel()

    '''
    plt.plot(img_diff_abs)
    plt.axhline(cut)
    plt.plot(img_diff_abs > cut)
    '''
    start = 0
    end = 0
    startHis = 0
    endHis = 0
    for coords in range(getCoords.shape[0]-1):
        if getCoords[coords+1] - getCoords[coords] > maxInterval:
            # print(getCoords[coords+1],getCoords[coords])
            start = coords
            end = coords
        else:
            end = coords+1
        if getCoords[end] - getCoords[start] >= getCoords[endHis] - getCoords[startHis]:
            startHis = start +1
            endHis = end
    if img_diff_abs[(getCoords[startHis] - bandWidth): getCoords[startHis]].shape[0] != 0 and  img_diff_abs[getCoords[endHis] : (getCoords[endHis] + bandWidth ) ].shape[0] != 0 :
        s = getCoords[startHis] + np.argmin(img_diff_abs[(getCoords[startHis] - bandWidth) : getCoords[startHis]]) - bandWidth
        e = getCoords[endHis] + np.argmin(img_diff_abs[getCoords[endHis] : (getCoords[endHis] + bandWidth ) ])
        return (s,e)
    else:
        return (getCoords[startHis],getCoords[endHis])



def getSplitPoint(img_diff):
    temp = np.arange(img_diff.shape[0])
    spl = UnivariateSpline(temp, img_diff)
    img_diff_abs = np.abs(spl(temp))


    # 加入平滑处理

    '''
    plt.plot(temp,img_diff)
    plt.plot(temp, spl(temp), 'g', lw=3)

    '''


    above_half_range = img_diff_abs[img_diff_abs > 0.3 * img_diff_abs.max()]
    height = above_half_range[np.logical_and(above_half_range < np.percentile(above_half_range, 80),
                                             above_half_range > np.percentile(above_half_range, 20))].mean()

    '''
    plt.plot(img_diff_abs)
    plt.axhline(height)
    '''

    ironRegion = []
    for alpha in np.linspace(1, 100, 10) / 1000:
        a, b = getIntervalOfIron(alpha * height, img_diff_abs)
        ironRegion.append(b - a)

    ironRegion = np.array(ironRegion)
    alpha = (np.linspace(1, 100, 10) / 1000)[np.argmin(np.diff(ironRegion)) + 1]
    a, b = getIntervalOfIron(alpha * height, img_diff_abs)
    return  a,b,alpha


def getImageSplitPoint(img,precision = 5,ifPlot = False,sides=10):
    row = img.shape[0]
    cutPoint =(np.linspace(0,1,precision+2)*row).astype(int)[1:-1]
    start = []
    end = []
    for cut in cutPoint:
        img_diff = (np.diff(img[cut:(cut + 80), :].mean(0)))
        a, b ,splitAlpha= getSplitPoint(img_diff)
        '''
        plt.plot(img_diff)
        plt.axvline(a)
        plt.axvline(b)
        '''
        start.append(a)
        end.append(b)
        if ifPlot:
            plt.plot(img_diff)
            plt.axvline(a,c = 'r')
            plt.axvline(b, c = 'r')
            plt.show()

    return int(np.median(a))-sides,int(np.median(b))+sides

def edgeCorrection(imgC):
    searchEdge = (np.linspace(0,1,5)*imgC.shape[0]).astype(int)[1:-1]
    rightList = []
    leftList = []
    for edgeSeaching in searchEdge:
        temp = cv2.cvtColor(imgC, cv2.COLOR_RGB2HSV)[edgeSeaching, :, 0]
        tempSplit  = 0.5*(temp.max()- temp.min()) + temp.min()
        '''
        plt.plot(temp)
        plt.axhline(tempSplit)
        '''
        part1 = np.argwhere(temp< tempSplit).ravel()
        part2 = np.argwhere(temp>tempSplit).ravel()
        part = part1 if part1.shape[0] < part2.shape[0] else part2

        # 找到右侧边缘
        # 未来找边缘算法应当改进为分位数操作
        try:
            right = part[part > (temp.shape[0]/2)].min()
        except:
            right = temp.shape[0]
        try:
            left  = part[part < (temp.shape[0]/2)].max()
        except:
            left = 0
        rightList.append(right)
        leftList.append(left)
    return max(int(np.median(leftList)) - 10,0),int( np.median(rightList)) + 10

