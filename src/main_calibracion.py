from libreria.constants import calibration
from libreria.funciones import calibrar_camara, numero_imgs_optimo

if __name__ == "__main__":
    intrinsics, dist_coeffs, rms, corners, chessboard_points = calibrar_camara(calibration)
    numero_imgs_optimo(corners, chessboard_points)

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)
