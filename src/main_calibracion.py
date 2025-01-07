from typing import List
import numpy as np
import cv2
import os
import copy
import matplotlib.pyplot as plt 
from libreria.constants import calibration

def get_chessboard_points(chessboard_shape, dx, dy):
    chessboard_points = np.zeros((chessboard_shape[0]*chessboard_shape[1], 3), np.float32)
    chessboard_points[:,:2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)
    chessboard_points[:, 0] *= dx
    chessboard_points[:, 1] *= dy
    return chessboard_points
 

def calibrar_camara(calibration):
    imgs = calibration
    corners = []
    ret_list = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for img in imgs:
        cor = cv2.findChessboardCorners(img, (8,6))
        corners.append(cor)

    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    corners_refined = [cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy = copy.deepcopy(imgs)

    imgs_corners = [cv2.drawChessboardCorners(img, (8,6), cor[1], cor[0]) if cor[0] else img for img, cor in zip(imgs_copy, corners)]
    os.makedirs("images/calibration/corners", exist_ok=True)
    for i in range(len(imgs_corners)):
        cv2.imwrite(f"images/calibration/corners/corners_{i}.jpg", imgs_corners[i])

    chessboard_points = get_chessboard_points((8, 6), 30, 30)

    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera([chessboard_points]*len(valid_corners), valid_corners, (8,6), None, None)
    # extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    return intrinsics, dist_coeffs, rms, corners, chessboard_points


def numero_imgs_optimo(corners, chessboard_points):
    # Calculate RMS for number of images
    image_counts = range(2, len(corners)+1)
    rms_errors = []
    for i in image_counts:
        valid_corners = [cor[1] for cor in corners[:i] if cor[0]]
        valid_corners = np.asarray(valid_corners, dtype=np.float32)
        rms, _, _, _, _ = cv2.calibrateCamera([chessboard_points]*len(valid_corners), valid_corners, (8,6), None, None)
        rms_errors.append(rms)

    # Plot 
    plt.figure(figsize=(10, 6))
    plt.plot(image_counts, rms_errors, marker='o')
    plt.title('RMS Reprojection Error vs. Number of Images')
    plt.xlabel('Number of Images Used')
    plt.ylabel('RMS Reprojection Error')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    intrinsics, dist_coeffs, rms, corners, chessboard_points = calibrar_camara(calibration)
    numero_imgs_optimo(corners, chessboard_points)

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)


