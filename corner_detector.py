import numpy as np
import cv2
import os
def check_valid_neighbors(im, i, j):
    valid_flag = False
    for x in range(i-1, i+2):
        for y in range(j-1, j+2):
            if x <0 or y < 0:
                continue
            try:
                if list(im[x][y]) != [0, 0, 0]:
                    valid_flag = True
            except:
                pass
    return valid_flag

def dist_calc(i, j, x, y):
    return (i-x) * (i-x) + (j-y) * (j-y)


def find_corners(im):
    curmin1 = max(im.shape[0], im.shape[1]) * max(im.shape[0], im.shape[1])
    imin = im.shape[0]
    jmin = im.shape[1]

    curmin2 = max(im.shape[0], im.shape[1]) * max(im.shape[0], im.shape[1])
    imax = 0
    jmax = 0

    curmin3 = max(im.shape[0], im.shape[1]) * max(im.shape[0], im.shape[1])
    imin1 = im.shape[0]
    jmax1 = 0

    curmin4 = max(im.shape[0], im.shape[1]) * max(im.shape[0], im.shape[1])
    imax1 = 0
    jmin1 = im.shape[1]
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if list(im[i][j]) == [0, 0, 0]:
                dist1 = dist_calc(i, j, 0, 0)
                #if i < imin or j < jmin:
                if dist1 < curmin1:
                    if check_valid_neighbors(im, i, j):
                        curmin1 = dist1
                        imin = i
                        jmin = j
                dist2 = dist_calc(i, j, im.shape[0]-1, im.shape[1]-1)
                #if i > imax or j > jmax:
                if dist2 < curmin2:
                    if check_valid_neighbors(im, i, j):
                        curmin2 = dist2
                        imax = i
                        jmax = j

    for j in range(im.shape[1]):
        for i in range(im.shape[0]):
            if list(im[i][j]) == [0, 0, 0]:
                dist3 = dist_calc(i, j, 0, im.shape[1]-1)
                #if i < imin1 or j > jmax1:
                if dist3 < curmin3:
                    if check_valid_neighbors(im, i, j):
                        curmin3 = dist3
                        imin1 = i
                        jmax1 = j
                dist4 = dist_calc(i, j, im.shape[0]-1, 0)
                if dist4 < curmin4:
                #if i > imax1 or j < jmin1:
                    if check_valid_neighbors(im, i, j):
                        curmin4 = dist4
                        imax1 = i
                        jmin1 = j

    if list(im[0][0]) != [0, 0, 0]:
        imin = 0
        jmin = 0

    if list(im[0][im.shape[1]-1]) != [0, 0, 0]:
        imin1 = 0
        jmax1 = im.shape[1]-1

    if list(im[im.shape[0]-1][0]) != [0, 0, 0]:
        imax1 = im.shape[0]-1
        jmin1 = 0

    if list(im[im.shape[0]-1][im.shape[1]-1]) != [0, 0, 0]:
        imax = im.shape[0]-1
        jmax = im.shape[1]-1

    return {"LT": (imin, jmin),
            "RT": (imin1, jmax1),
            "LB": (imax1, jmin1),
            "RB": (imax, jmax)}

# for i in os.listdir('results'):
#     print(i)
img = cv2.imread('./results/0_output.png')
corners = find_corners(img)
print(corners)
arr = []
for i in sorted(corners.keys()):
    print(i)
    arr.append((corners[i][1],corners[i][0]))
print(arr)
final = []
LTX = min(corners['LT'][1],corners['LB'][1])
LTY = min(corners['LT'][0],corners['LB'][0])
LBX = LTX
RTY = LTY
RBX = max(corners['RT'][1],corners['RB'][1])
RBY = max(corners['RT'][0],corners['RB'][0])
LBY = RBY
RTX = RBX
final.append((LBX,LBY))
final.append((LTX,LTY))
final.append((RBX,RBY))
final.append((RTX,RTY))
print(final)
pts1 = np.float32(arr)
pts2 = np.float32(final)
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (RBX, RBY))

cv2.imwrite('result.jpg',result)
# break
