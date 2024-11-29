import cv2
import numpy as np
import matplotlib.pyplot as plt
from libreria.utils import *
from libreria.bow import BoW
from libreria.dataset import Dataset
from libreria.image_classifier import ImageClassifier
import time
from tqdm import tqdm
import sys
import pickle
from libreria.funciones_descriptores import *

if __name__ == "__main__":
    # Rutas de las imágenes
    # coge las imagenes que hay en figures

    reference_images = [
        "./images/figures/triangulo.png",
        "./images/figures/circulo.png",
        "./images/figures/rombo.png",
        "./images/figures/cuadrado.png",
    ]
    input_image = "./images/output/masks/mask1.png"
    reference_images = [
        cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in reference_images
    ]
    input_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2GRAY)
    # show_image(reference_images)
    # show_image([input_image])

    # Inicializar ORB y BFMatcher
    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Comparar imágenes
    best_match_index, match_count = compare_images(
        input_image, reference_images, detector, matcher
    )

    if best_match_index != -1:
        print(
            f"La imagen más similar es: {reference_images[best_match_index]} con {match_count} coincidencias"
        )
    else:
        print("No se encontraron coincidencias significativas.")
