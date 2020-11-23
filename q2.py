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
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1000)
    kp1_SIFT, desc1_SIFT = sift.detectAndCompute(img1, None)
    kp2_SIFT, desc2_SIFT = sift.detectAndCompute(img2, None)
    kp1_SURF, desc1_SURF = surf.detectAndCompute(img1, None)
    kp2_SURF, desc2_SURF = surf.detectAndCompute(img2, None)
    kp1_ORB, desc1_ORB = orb.detectAndCompute(img1, None)
    kp2_ORB, desc2_ORB = orb.detectAndCompute(img2, None)
    img1_SIFT = cv2.drawKeypoints(img1, kp1_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,0))
    img2_SIFT = cv2.drawKeypoints(img2, kp1_SIFT, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,0))
    
    img1_SURF = cv2.drawKeypoints(img1, kp1_SURF, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(150,0,0))
    img2_SURF = cv2.drawKeypoints(img2, kp1_SURF, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(150,0,0))
    img1_ORB = cv2.drawKeypoints(img1, kp1_ORB, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(150,0,0))
    img2_ORB = cv2.drawKeypoints(img2, kp1_ORB, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(150,0,0))
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
        pts1 = np.float32([[src_pts[random_list[0]][0], src_pts[random_list[0]][1]],
        [src_pts[random_list[1]][0], src_pts[random_list[1]][1]],
        [src_pts[random_list[2]][0], src_pts[random_list[2]][1]]])
        pts2 = np.float32([[dst_pts[random_list[0]][0], dst_pts[random_list[0]][1]],
        [dst_pts[random_list[1]][0], dst_pts[random_list[1]][1]],
        [dst_pts[random_list[2]][0], dst_pts[random_list[2]][1]]])
        A = cv2.getAffineTransform(pts1, pts2)
        m, n = dst_pts.shape
        threshold = 5
        count = 0
        for j in range(len(src_pts)):
            buff_pt = np.zeros((3,),dtype="float")
            buff_pt[0] = src_pts[i][0]
            buff_pt[1] = src_pts[i][1]
            buff_pt[2] = 1.0
            buff_pt.reshape((3,1))
            new_pt = A.dot(buff_pt.T)
            if math.sqrt((new_pt[0]-dst_pts[j][0])**2 + (new_pt[1]-dst_pts[j][1])**2)<5: count += 1
        if count > precount:
            precount = count
            best_A = A
            
    cols, rows = img1.shape
    img3=cv2.warpAffine(img1, best_A, (cols, rows))
    plt.imshow(img3),plt.show()
    
    
if __name__ == '__main__':
    match("./Book_cover.jpg", "./Book_pic.png")
