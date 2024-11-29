import cv2
import numpy as np
import matplotlib.pyplot as plt
from Lab_Project.src.libreria.utils import *
from Lab_Project.src.libreria.bow import BoW
from Lab_Project.src.libreria.dataset import Dataset
from Lab_Project.src.libreria.image_classifier import ImageClassifier
import time
from tqdm import tqdm
import sys
import pickle
from Lab_Project.src.libreria.funciones_descriptores import *

if __name__ == "__main__":
    # Rutas de las imágenes
    # coge las imagenes que hay en figures

    reference_images = [
        "figures/triangulo.png",
        "figures/circulo.png",
        "figures/rombo.png",
        "figures/cuadrado.png",
    ]
    input_image = "masks/mask2.png"
    reference_images = [
        cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in reference_images
    ]
    input_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2GRAY)
    show_image(reference_images)
    show_image([input_image])

    # Inicializar ORB y BFMatcher
    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
