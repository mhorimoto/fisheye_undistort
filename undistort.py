#! /usr/bin/env python3
#
# You should replace these 3 lines with the output in calibration step
#
import cv2
import numpy as np
import sys
import os

SAVE_FOLDER_PATH = "./result/"
DIM=(2592, 1944)
K=np.array([[924.2489622910588, 0.0, 1341.350075959198], [0.0, 924.4718676645766, 1024.8809192234582], [0.0, 0.0, 1.0]])
D=np.array([[0.04042287155645841], [0.009335117902960724], [-0.012746667178224234], [0.003051296091374329]])
nK=np.array([[1157.23313071166785,0.0,1296.58960781949304],[0.0,1152.84203305645997,968.88919877676904],[0.0,0.0,1.0]])
nD=np.array([[-0.27704417675051],[-0.13422693560528],[0.01784702166394],[0.00657667226218],[0.13855670364354]])

def saveImg(dirPath, fname, img,addf=""):
    f = os.path.basename(fname).split('.', 1)[0]
    path = dirPath + f + addf + "_res.png"
    cv2.imwrite(path,img)
    print("saved: ",path)

def undistort_nofish(img_path):
    img = cv2.imread(img_path)
    resultImg = cv2.undistort(img, nK, nD, None) # 内部パラメータを元に画像補正
    saveImg(SAVE_FOLDER_PATH,img_path, resultImg,"_nofish")

def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    saveImg(SAVE_FOLDER_PATH,img_path,undistorted_img)
#    cv2.imshow("undistorted", undistorted_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def undistort2(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    appendfn = "_balance{0}".format(balance)
    saveImg(SAVE_FOLDER_PATH,img_path,undistorted_img,appendfn)
#    cv2.imshow("undistorted", undistorted_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort2(p,0.0)
        undistort2(p,0.3)
        undistort2(p,0.8)
        undistort2(p,1.0)
        undistort(p)
        undistort_nofish(p)
    quit()
    
