import os
import glob
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp
from numpy.linalg import norm
from stableMethod.utils import *
import sys
import json

if __name__ == "__main__":

    # 先进行gamma变换处理

    img = cv2.imread("../img/sample/sample4.jpg")
    ifVertical = True
    if ifVertical:
        img = img.transpose(1,0,2)
    # img = cv2.imread("copper.jpg")
    # 添加调整图像水平的代码
    dst = derectionCorrect(img)
    plotCorrect = False
    if plotCorrect:
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(dst)
    plt.show()
    img = dst
    # hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    # hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    # hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    img_corrected = gamma_trans(img, 0.5)
    # cv2.imwrite('gamma_corrected_iron.png', img_corrected)

    # 跳过分类器部分，直接对于铜片进行检测
    imgC = img_corrected.copy()
    # imgC = cv2.imread("gamma_corrected_iron.png")
    imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
    # 取出蓝色通道
    # img  = cv2.cvtColor(imgC,cv2.COLOR_BGR2HSV)[...,0]
    # img = cv2.imread("gamma_corrected_iron.png", 0)

    b, g, r = cv2.split(imgC)

    # img = (g.astype(np.float32) + r.astype(np.float32)) / (b.astype(np.float32) + 10)
    img = cv2.cvtColor(imgC,cv2.COLOR_RGB2GRAY)

    img = img.transpose()
    imgC = imgC.transpose(1, 0, 2)


    manual = False
    if manual:
        # 交互式的获取铜片的位置，并且进行拉伸变换
        plt.imshow(imgC)
        pos = plt.ginput(4)  # 依照顺序框出四个点
        plt.close()
        pos = np.array([[i[0], i[1]] for i in pos])
        xstart = int(pos[:, 0].min())
        xend = int(pos[:, 0].max())
        ystart = int(pos[:, 1].min())
        yend = int(pos[:, 1].max())

        imgC = imgC[ystart:yend, xstart:xend]
        img = img[ystart:yend, xstart:xend]

    else :
        # 根据 getIntervalOfIron 确认边缘信息
        start, end = getImageSplitPoint(img)
        row = img.shape[0]
        img  = img[int(row * 0.2):int( row * 0.8) ,start:end]
        imgC = imgC[int(row * 0.2):int(row * 0.8), start:end]

        # 注意：目前的edgeCorrection 不能很好的处理垫片！！
        start , end = edgeCorrection(imgC)
        img = img[:, start:end]
        imgC = imgC[:, start:end]

    plt.imshow(imgC)

    # 修改内容：添加了对于区域的自适应的范围选择
    rowLength = imgC.shape[0]
    rowStart = int(rowLength * 0.2)
    rowEnd = int(rowLength * 0.8)

    rowList = np.array(range(rowStart, rowEnd, 10))
    np.random.shuffle(rowList)

    # rowList = [750]
    counter = []
    lr = None
    plotNum = 0
    plotCounter = 0

    #
    #
    # for startRow in rowList:
    #     cnt = showResult(lr, startRow=startRow, rowNum=10,
    #                  plot=True, argrelmaxOrder=40, d=3, scale=1.15
    #                  )
    #
    #     print(cnt)
    #
    #     counter.append(cnt)
    #
    # print(counter)
    #
    # from   collections import Counter
    #
    # print(Counter(counter))
    #
    #
    #
    nrowList = rowList.shape[0]
    rowLists = np.split(rowList, [
        int(0.2 * nrowList),
        int(0.4 * nrowList),
        int(0.6 * nrowList),
        int(0.8 * nrowList)
    ])

    counter = []
    referenceInfo = []
    lr = None
    plotNum = 0
    plotCounter = 0

    # rowLists = [[773,703,943,1203,1173]]
    for rowlist in rowLists:
        for startRow in rowlist:
            cnt = showResult(lr,img,imgC ,plotNum,plotCounter,startRow=startRow, rowNum=10,
                             plot=False, argrelmaxOrder=20, d=3, scale=1.15
                             )

            # print(cnt)
            counter.append(cnt['counter'])
            referenceInfo.append(cnt)

        # 判断counter 的计数结果，确认是否是合理的
        from collections import Counter

        cnter = dict(Counter(counter))
        cnterResult = max(cnter, key=cnter.get)
        prob = cnter[cnterResult] / len(counter)
        # print(prob)
        # print(cnter)

        if prob > 0.8:
            break




    from collections import Counter
    count = sorted(Counter(counter).items(),key = lambda x:x[1],reverse = True)[0][0]
    pvalue = max(Counter(counter).values()) / counter.__len__()

    # 选择部分数据进行绘图
    referenceForPlot = [record for record in referenceInfo if record['reference']['xcoordsLen'] == count][:3]
    floder = os.path.dirname(os.path.dirname(__file__))
    for plotNum , reference in enumerate(referenceForPlot):
        fig = plt.figure(figsize=(20, 3))
        plt.title("counter : " + str(count))
        print((imgC[reference['reference']['startRow']:(reference['reference']['startRow'] + 40), :]).shape)
        plt.imshow(imgC[reference['reference']['startRow']:(reference['reference']['startRow']+ 40),:])
        for xc in reference['reference']['xcoords']:
            plt.axvline(x=xc, color='r', linewidth=4)
        plt.savefig(os.path.join(floder, "output\\results_" + str(plotNum) + ".jpg"), dpi=500)
        plt.close()

    # 将结果合并
    floder = os.path.dirname(os.path.dirname(__file__))
    for i in range(IMAGE_NEED):
        exec("r"+str(i)+" = cv2.imread(os.path.join('" + floder +"', 'output//results_"+str(i) + ".jpg'))")

    r = eval("np.concatenate([" + ','.join([ j+str(i) for i,j in enumerate(['r']*IMAGE_NEED)]) + "])")
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(floder,"output/results.png"),r)


    tempFile = glob.glob(os.path.join(floder,"output/results_*"))

    for file in tempFile :
       os.remove(file)

    message = json.dumps({"计数": count, "rate": pvalue}, ensure_ascii=False)
    print( {"result_img": r, "message": message, "class": "pcb_count", "save": False})

    #
    #
    # # 将结果合并
    floder = os.path.dirname(os.path.dirname(__file__))
    print(floder)
    floder = "C:/Users/an/PycharmProjects/ironCounterProject"

    # for i in range(IMAGE_NEED):
    #     exec("r"+str(i)+" = cv2.imread(os.path.join('" + floder +"', 'output//results_"+str(i) + ".jpg'))")
    #
    # r = eval("np.concatenate([" + ','.join([ j+str(i) for i,j in enumerate(['r']*IMAGE_NEED)]) + "])")
    # r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(floder,"output/results.png"),r)
    #
    #
    # tempFile = glob.glob(os.path.join(floder,"output/results_*"))
    # #for file in tempFile :
    # #    os.remove(file)
    # message = json.dumps({"计数": count, "rate":pvalue}, ensure_ascii=False)
    # # return { "result_img" : r,"message": message,"class":"pcb_count","save":False}
