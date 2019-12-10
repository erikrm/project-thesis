import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as pyplot

# Load functions:

def list_all_files_in_folder(folder_path, reference_name, file_type):
    file_paths = []
    file_names = []

    for filename in os.listdir(folder_path):

        if filename.endswith(file_type):
            if reference_name in filename:
                reference_path = os.path.join(folder_path, filename)
            else:
                full_path = os.path.join(folder_path, filename)
                file_paths.append(full_path)
                file_names.append(filename.replace(file_type,''))

    return file_paths, file_names, reference_path

def load_images(file_paths, reference_path):
    reference_image = cv2.imread(reference_path)
    images_shape = (len(file_paths),) + reference_image.shape
    images = np.zeros(images_shape, dtype=reference_image.dtype)

    i = 0
    for file_path in file_paths:
        images[i] = cv2.imread(file_path)

    return images, reference_image

def load_spectrums(file_paths, reference_path):
    
    dtype = 'float64' #TODO: change to automatical type?

    dataset_length = len(spectrum_paths)

    spectrum_reference = np.loadtxt(reference_path, dtype=dtype, skiprows=17)

    shape_spectrums = (dataset_length,) + spectrum_reference.shape
    spectrums = np.zeros(shape=shape_spectrums, dtype=dtype)

    for i in range(0,dataset_length):
        spectrums[i] = np.loadtxt(spectrum_paths[i], skiprows=17, dtype=dtype)

    return spectrums, spectrum_reference

def load_qe(qe_path):
    qe_angle_blue = np.loadtxt(qe_path, skiprows=2)

    # Constants that shouldn't be here, but I can't bother to pass them down:
    freq_start = 300
    freq_end = 1100
    lambda_total = freq_end - freq_start
    px_total_horizontal = 1320
    
    qe_total = 0.5
    px_total_vertical = 624

    #Retrieve x-axis, given by x = sin(angle) * Px * (lambda_total)/px_total_horizontal

    x_vector = np.cos(qe_angle_blue[:,1]/180*np.pi) * qe_angle_blue[:,0] *lambda_total/px_total_horizontal + freq_start
    y_vector = np.sin(qe_angle_blue[:,1]/180*np.pi) * qe_angle_blue[:,0] * qe_total/px_total_vertical

    spectrum_qe = np.array([x_vector, y_vector])
    return spectrum_qe

def load_qe_all_colors(qe_paths):
    spectrum_qe_blue = load_qe(qe_paths[0])
    spectrum_qe_green = load_qe(qe_paths[1])
    spectrum_qe_red = load_qe(qe_paths[2])

    return [spectrum_qe_blue, spectrum_qe_green, spectrum_qe_red]


# Visualization functions:
def normalize(array):
    amax = np.amax(array)
    #amin = 0 
    #np.amin(array)
    return np.divide(array,amax)

def histogram_equalization(array):
    array_norm = normalize(array)
    return np.array(np.round(array_norm*255), dtype="uint8")


if __name__ == "__main__":
    # Variables:
    image_folder_path = ".\\figures\\camera_pictures\\"
    image_file_type = ".bmp"
    image_reference_name = "001_background"

    spectrum_folder_path = ".\\spectrum_files\\"
    spectrum_file_type = ".txt"
    spectrum_reference_name = "001_background"
    qe_paths = [".\\qe_spectrum\\QE_angle_blue.txt", ".\\qe_spectrum\\QE_angle_green.txt", ".\\qe_spectrum\\QE_angle_red.txt"]

    # Find images
    image_paths, image_names, image_reference_path = list_all_files_in_folder(image_folder_path, image_reference_name,image_file_type)

    # Find spectrums
    spectrum_paths, spectrum_names, spectrum_reference_path = list_all_files_in_folder(spectrum_folder_path, spectrum_reference_name, spectrum_file_type)

    if spectrum_names != image_names:
        print("Not all images have spectrums or vise versa")

    # Load images
    images, image_reference = load_images(image_paths, image_reference_path)

    # Load spectrums
    spectrums, spectrum_reference = load_spectrums(spectrum_paths, spectrum_reference_path)

    # Load quantum efficiency
    qe_spectrums = load_qe_all_colors(qe_paths)
    
