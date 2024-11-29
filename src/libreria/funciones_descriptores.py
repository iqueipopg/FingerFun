import cv2
import numpy as np
import matplotlib.pyplot as plt
from libreria.utils import *
from libreria.bow import BoW
from libreria.dataset import Dataset
from libreria.image_classifier import ImageClassifier


def show_image(imgs, num=None):
    if num is not None:
        imgs = imgs[:num]

    for img in imgs:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def generateGaussianImages(image, sigmas):
    """Generate the gaussian images using the base image and the sigmas given

    Args:
        image (np.array[np.float32]): Base image to blur
        sigmas (List[np.float32]): Sigmas for blurring the image

    Returns:
        List[np.array[np.float32]: List of blurred images
    """
    gaussian_images = []

    # TODO: Generate the list of blurred images using cv2.GaussianBlur()
    for sigma in sigmas:

        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        gaussian_images.append(image)

    return gaussian_images


def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussians list

    Args:
        gaussian_images (List[np.array[np.float32]): List of blurred images

    Returns:
        List[np.array[np.float32]: List of difference of gaussian images
    """
    dog_images = []

    # TODO: Generate the list of difference of gaussians using cv2.subtract()

    for i in range(1, len(gaussian_images)):
        dog_img = cv2.subtract(gaussian_images[i], gaussian_images[i - 1])
        # cv2.imshow(f"DOG {i}", dog_img.astype(np.uint8))
        dog_images.append(dog_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dog_images


def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 array composed of subimages
    is strictly greater than or less than all its neighbors, False otherwise.

    Args:
        first_subimage (np.array): Patch from first DoG
        second_subimage (np.array): Patch from second DoG (center layer)
        third_subimage (np.array): Patch from third DoG
        threshold (float): Value threshold for the pixel

    Returns:
        Bool: True if maximum or minimum, False otherwise
    """
    # Obtener el valor del píxel central
    center_pixel = second_subimage[1, 1]
    # Crear una lista con todos los valores vecinos en el cubo 3x3x3
    neighbors = np.concatenate(
        [first_subimage.flatten(), second_subimage.flatten(), third_subimage.flatten()]
    )

    # print(first_subimage.shape, second_subimage.shape, third_subimage.shape)

    # Remover el píxel central de los vecinos
    neighbors = np.delete(neighbors, 13)

    # Comprobar si el píxel central es un máximo o mínimo absoluto y cumple el umbral

    is_maximum = np.all(center_pixel >= neighbors)
    is_minimum = np.all(center_pixel <= neighbors)

    # Check if the center pixel meets the threshold condition
    if is_maximum or (is_minimum and (center_pixel < -threshold)):
        return True

    return False


def findScaleSpaceExtrema(
    gaussian_images, dog_images, num_intervals, sigma, threshold=0.03
):
    """Find pixel positions of all scale-space extrema in the image pyramid"""
    keypoints = []

    # Iteramos sobre los índices de las imágenes en diferencias de gaussianas

    # TODO: Fill the loop source data
    for image_index, (first_image, second_image, third_image) in enumerate(
        zip(dog_images, dog_images[1:], dog_images[2:])
    ):
        # (i, j) is the center of the 3x3 array
        # TODO: Fill the 2 range limits knowing you have to move the 3x3 window across the whole image
        for i in range(1, first_image.shape[0] - 1):
            for j in range(1, first_image.shape[1] - 1):
                # TODO: Fill the method with the required arguments
                if isPixelAnExtremum(
                    first_image[i - 1 : i + 2, j - 1 : j + 2],
                    second_image[i - 1 : i + 2, j - 1 : j + 2],
                    third_image[i - 1 : i + 2, j - 1 : j + 2],
                    threshold,
                ):
                    # Refine the keypoint localization
                    localization_result = localizeExtremumViaQuadraticFit(
                        i, j, image_index + 1, num_intervals, dog_images, sigma
                    )
                    if localization_result is not None:
                        keypoint, localized_image_index = localization_result
                        # Get the keypoint orientation
                        keypoints_with_orientations = computeKeypointsWithOrientations(
                            keypoint, gaussian_images[localized_image_index]
                        )
                        for keypoint_with_orientation in keypoints_with_orientations:
                            keypoints.append(keypoint_with_orientation)
    return keypoints


def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3):
    """Compute SIFT keypoints and descriptors for an input image"""
    # TODO: Fill the pipeline to get the keypoint and descriptors as before
    # image = image.astype('float32')
    image = image.astype(np.float32)
    gaussian_kernels = generateGaussianSigmas(sigma, num_intervals)
    gaussian_images = generateGaussianImages(image, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    # show_image(dog_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors


def match_descriptors(descriptors1, descriptors2, matcher):
    """
    Encuentra las coincidencias entre dos conjuntos de descriptores usando un matcher.
    """
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Ordenar por distancia
    return matches


def compare_images(input_image, reference_images, detector, matcher):
    """
    Compara una imagen de entrada con varias imágenes de referencia usando keypoints y descriptores.
    """
    # Extraer keypoints y descriptores de la imagen de entrada
    _, input_descriptors = computeKeypointsAndDescriptors(input_image)

    input_descriptors = input_descriptors.astype(np.float32)
    print(input_descriptors.shape)
    print("input_descriptors", len(input_descriptors))
    best_match_count = 0
    best_match_index = -1

    # Iterar sobre las imágenes de referencia
    for i, ref_image in enumerate(reference_images):
        _, ref_descriptors = computeKeypointsAndDescriptors(ref_image)
        print(ref_descriptors.shape)
        print(f"Tipo de descriptor1: {input_descriptors.dtype}")
        ref_descriptors = ref_descriptors.astype(np.float32)
        print(f"Tipo de descriptor2: {ref_descriptors.dtype}")
        print("ref_descriptors", len(ref_descriptors))

        matches = match_descriptors(input_descriptors, ref_descriptors, matcher)

        # Filtrar coincidencias con un umbral de calidad
        good_matches = [
            m for m in matches if m.distance < 50
        ]  # Ajustar el umbral según el detector
        print(f"Imagen {i+1}: {len(good_matches)} coincidencias buenas")

        # Guardar la mejor coincidencia
        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_match_index = i

    return best_match_index, best_match_count
