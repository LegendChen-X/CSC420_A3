import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math

def match(path_1, path_2):
    bf = cv2.BFMatcher()
    img1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1_SIFT, desc1_SIFT = sift.detectAndCompute(img1, None)
    kp2_SIFT, desc2_SIFT = sift.detectAndCompute(img2, None)
    img1_SIFT = cv2.drawKeypoints(img1, kp1_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,0))
    img2_SIFT = cv2.drawKeypoints(img2, kp1_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,0))
    kp1 = kp1_SIFT
    kp2 = kp2_SIFT
    desc1 = desc1_SIFT
    desc2 = desc2_SIFT
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    good_matches_without_list = []
    
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            good_matches_without_list.append(m)
    precount = 0
    iters = 40
    best_A = None
    P = 0.995
    for i in range(iters):
        random_list = []
        while len(random_list)!=3:
            i = random.randrange(0, len(good_matches))
            if i not in random_list: random_list.append(i)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_without_list]).reshape(-1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_without_list]).reshape(-1,2)
# Get three random points from src
        pts1 = np.float32([[src_pts[random_list[0]][0], src_pts[random_list[0]][1]], [src_pts[random_list[1]][0], src_pts[random_list[1]][1]], [src_pts[random_list[2]][0], src_pts[random_list[2]][1]]])
# Get three random points from dst
        pts2 = np.float32([[dst_pts[random_list[0]][0], dst_pts[random_list[0]][1]], [dst_pts[random_list[1]][0], dst_pts[random_list[1]][1]], [dst_pts[random_list[2]][0], dst_pts[random_list[2]][1]]])
# Compute AffineTransform from src to dst
        A = cv2.getAffineTransform(pts1, pts2)
        m, n = src_pts.shape
        threshold = 5
        count = 0
        for j in range(m):
            buff_pt = np.zeros((3,),dtype="float")
            buff_pt[0] = src_pts[j][0]
            buff_pt[1] = src_pts[j][1]
            buff_pt[2] = 1.0
            buff_pt.reshape((3,1))
            new_pt = A.dot(buff_pt.T)
            if (abs(new_pt[0]-dst_pts[j][0]) + abs(new_pt[1]-dst_pts[j][1])) < 5: count += 1
        if count > precount:
            precount = count
            best_A = A
    print(best_A)
    h,w = img1.shape
    new_pt_1 = best_A.dot(np.array([0,0,1]).T)
    new_pt_2 = best_A.dot(np.array([w-1,0,1]).T)
    new_pt_3 = best_A.dot(np.array([0,h-1,1]).T)
    new_pt_4 = best_A.dot(np.array([w-1,h-1,1]).T)
    detected_book = np.array([new_pt_1,new_pt_3,new_pt_4,new_pt_2])
    img3 = cv2.polylines(img2,[np.int32(detected_book)], True, 255, 3, cv2.LINE_AA)
    plt.imshow(img3, 'gray')
    plt.show()
    img4 = cv2.drawMatchesKnn(img1,kp1,img3,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))
    plt.imshow(img4)
    plt.show()
    
    
def dynamtic_map(path_1, path_2, estimate):
    bf = cv2.BFMatcher()
    img1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1_SIFT, desc1_SIFT = sift.detectAndCompute(img1, None)
    kp2_SIFT, desc2_SIFT = sift.detectAndCompute(img2, None)
    img1_SIFT = cv2.drawKeypoints(img1, kp1_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,0))
    img2_SIFT = cv2.drawKeypoints(img2, kp1_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,0))
    kp1 = kp1_SIFT
    kp2 = kp2_SIFT
    desc1 = desc1_SIFT
    desc2 = desc2_SIFT
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    good_matches_without_list = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            good_matches_without_list.append(m)
    precount = 0
    iters = 9999999999999
    best_A = None
    P = 0.995
    for i in range(iters):
        random_list = []
        while len(random_list)!=3:
            i = random.randrange(0, len(good_matches))
            if i not in random_list: random_list.append(i)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_without_list]).reshape(-1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_without_list]).reshape(-1,2)
# Get three random points from src
        pts1 = np.float32([[src_pts[random_list[0]][0], src_pts[random_list[0]][1]], [src_pts[random_list[1]][0], src_pts[random_list[1]][1]], [src_pts[random_list[2]][0], src_pts[random_list[2]][1]]])
# Get three random points from dst
        pts2 = np.float32([[dst_pts[random_list[0]][0], dst_pts[random_list[0]][1]], [dst_pts[random_list[1]][0], dst_pts[random_list[1]][1]], [dst_pts[random_list[2]][0], dst_pts[random_list[2]][1]]])
# Compute AffineTransform from src to dst
        A = cv2.getAffineTransform(pts1, pts2)
        m, n = src_pts.shape
        threshold = 5
        count = 0
        for j in range(m):
            buff_pt = np.zeros((3,),dtype="float")
            buff_pt[0] = src_pts[j][0]
            buff_pt[1] = src_pts[j][1]
            buff_pt[2] = 1.0
            buff_pt.reshape((3,1))
            new_pt = A.dot(buff_pt.T)
            if (abs(new_pt[0]-dst_pts[j][0]) + abs(new_pt[1]-dst_pts[j][1])) < 5: count += 1
        if (count / m) > estimate:
            iters = int(math.log(1 - P) / math.log(1 - pow(count / m, 3)))
            estimate = (count / m)
            print(iters)
        if count > precount:
            precount = count
            best_A = A
        if count > 0.995 * estimate * m : break
    h,w = img1.shape
    new_pt_1 = best_A.dot(np.array([0,0,1]).T)
    new_pt_2 = best_A.dot(np.array([w-1,0,1]).T)
    new_pt_3 = best_A.dot(np.array([0,h-1,1]).T)
    new_pt_4 = best_A.dot(np.array([w-1,h-1,1]).T)
    detected_book = np.array([new_pt_1,new_pt_3,new_pt_4,new_pt_2])
    img3 = cv2.polylines(img2,[np.int32(detected_book)], True, 255, 3, cv2.LINE_AA)
    plt.imshow(img3, 'gray')
    plt.show()
    img4 = cv2.drawMatchesKnn(img1,kp1,img3,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,255,0))
    plt.imshow(img4)
    plt.show()
    
if __name__ == '__main__':
    dynamtic_map("./Book_cover.jpg", "./Book_pic.png", 0.25)
