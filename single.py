import cv2
import numpy as np
import os

path = "C:/Users/Furkan/Desktop/maping/frames"

def combining(directory):
    sift = cv2.SIFT_create()

    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    image_files = [f for f in os.listdir(
        directory) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort()
    print(image_files)
    img1 = cv2.imread(os.path.join(directory, image_files[0]))

    for image_file in image_files[1:]:
        img2 = cv2.imread(os.path.join(directory, image_file))

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is not None and des2 is not None and des1.dtype == des2.dtype:
            matches = bf.match(des1, des2)
        else:
            print("Descriptors are either empty or have different types.")
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 1:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            points = np.float32(
                [[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(points, M)
            raw_points = np.float32(
                [[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            all_points = np.concatenate((dst, raw_points), axis=0)

            [xmin, ymin] = np.int32(all_points.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_points.max(axis=0).ravel() + 0.5)
            tform_width = xmax - xmin
            tform_height = ymax - ymin

            tform = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
            panorama = cv2.warpPerspective(
                img1, tform.dot(M), (tform_width, tform_height))

            xstart = max(-xmin, 0)
            ystart = max(-ymin, 0)
            xend = min(xstart+img2.shape[1], panorama.shape[1])
            yend = min(ystart+img2.shape[0], panorama.shape[0])

            panorama[ystart:yend, xstart:xend] = img2

            img1 = panorama

        else:
            print("Not enough matches are found between", img1, "and", img2)

    #cv2.imshow("Final Panorama", panorama)
    cv2.imwrite('final_results.jpg', panorama)

combining(path)
cv2.waitKey(0)
cv2.destroyAllWindows()
