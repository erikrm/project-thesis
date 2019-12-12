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
        i = i + 1

    return images, reference_image

def load_spectrums(file_paths, reference_path):
    
    dtype = 'float64' #TODO: change to automatical type?

    dataset_length = len(file_paths)

    spectrum_reference = np.loadtxt(reference_path, dtype=dtype, skiprows=17)

    shape_spectrums = (dataset_length,) + spectrum_reference.shape
    spectrums = np.zeros(shape=shape_spectrums, dtype=dtype)

    for i in range(0,dataset_length):
        spectrums[i] = np.loadtxt(file_paths[i], skiprows=17, dtype=dtype)

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
    return np.swapaxes(spectrum_qe, axis1=0, axis2=1)

def load_qe_all_colors(qe_paths):
    spectrum_qe_blue = load_qe(qe_paths[0])
    spectrum_qe_green = load_qe(qe_paths[1])
    spectrum_qe_red = load_qe(qe_paths[2])

    # Since they have irregular length it is easiest to keep them seperated until interpolated, I want to avoid mixing python list and numpy array
    return spectrum_qe_blue, spectrum_qe_green, spectrum_qe_red

# Calculations
def hadamard_division(matrix_divident, matrix_divisor):
    return np.divide(matrix_divident, matrix_divisor)

def interpolate_qe(spectrum_x, spectrum_qe_blue, spectrum_qe_green, spectrum_qe_red):
    left_value = 0.0
    right_value = 0.0

    if np.all(np.diff(spectrum_x)<0):
        print("Spectrum_x is not monotonic increasing")
        sys.exit(0)
    elif np.all(np.diff(spectrum_qe_blue[:,0])<0):
        print("Spectrum qe blue is not monotonic increasing")
        sys.exit(0)
    elif np.all(np.diff(spectrum_qe_green[:,0])<0):
        print("Spectrum qe green is not monotonic increasing")
        sys.exit(0)
    elif np.all(np.diff(spectrum_qe_red[:,0])<0):
        print("Spectrum qe red is not monotonic increasing")
        sys.exit(0)
    else:
        qe_interpolated_shape = spectrum_x.shape + (3,)
        qe_interpolated = np.zeros(shape=qe_interpolated_shape, dtype=spectrum_qe_blue.dtype)

        qe_interpolated[:,0] = np.interp(spectrum_x, spectrum_qe_blue[:,0], spectrum_qe_blue[:,1], left=left_value, right=right_value)
        qe_interpolated[:,1] = np.interp(spectrum_x, spectrum_qe_green[:,0], spectrum_qe_green[:,1], left=left_value, right=right_value)
        qe_interpolated[:,2] = np.interp(spectrum_x, spectrum_qe_red[:,0], spectrum_qe_red[:,1], left=left_value, right=right_value)

        return qe_interpolated

# Visualization functions:
def normalize(array):
    amax = np.amax(array)
    #amin = 0 
    #np.amin(array)
    return np.divide(array,amax)

def histogram_equalization(array):
    array_norm = normalize(array)
    return np.array(np.round(array_norm*255), dtype="uint8")

def hadamard_two_face(matrix_object, matrix_reference, dark_limit):
    RR_negative = np.zeros(np.shape(matrix_object), dtype='float64')
    RR_positive = np.array(RR_negative)

    RR_negative = (matrix_reference > matrix_object + dark_limit)*(np.divide(matrix_reference, matrix_object))
    RR_positive = (matrix_object > matrix_reference + dark_limit)*(np.divide(matrix_object, matrix_reference))

    # This function readies images for visualization, therefore it doesn't matter that we are removing information by histogram equalization
    RR_negative_int = histogram_equalization(RR_negative)
    RR_positive_int = histogram_equalization(RR_positive)

    return RR_negative_int, RR_positive_int

def wait_user_input():    
    break_all = False
    while(True):
        key = cv2.waitKey(10000)
        if key == 27:
            break_all = True
            break  # esc to quit
        if key == ord('c'):
            break
    return break_all

def show_image(title, image):
    cv2.imshow(title,image)

def show_image_vector(titles, images):
    i = 0
    for image in images:
        show_image(titles[i], image)
        i = i + 1
        if wait_user_input():
            break

def plot_bgr(title, spectrum_bgr, x_axis):
    pyplot.title(title)
    pyplot.plot(x_axis, spectrum_bgr[:,0], color="Blue")
    pyplot.plot(x_axis, spectrum_bgr[:,1], color="Green")
    pyplot.plot(x_axis, spectrum_bgr[:,2], color="Red")


def main():
    # Variables:
    image_folder_path = ".\\figures\\camera_pictures\\"
    image_file_type = ".bmp"
    image_reference_name = "001_background"
    image_dark_limit = 10

    spectrum_folder_path = ".\\spectrum_files\\"
    spectrum_file_type = ".txt"
    spectrum_reference_name = "001_background"
    qe_paths = [".\\qe_spectrum\\QE_angle_blue.txt", ".\\qe_spectrum\\QE_angle_green.txt", ".\\qe_spectrum\\QE_angle_red.txt"]

    # Find images
    image_paths, image_names, image_reference_path = list_all_files_in_folder(image_folder_path, image_reference_name,image_file_type)

    # Find spectrums
    spectrum_paths, spectrum_names, spectrum_reference_path = list_all_files_in_folder(spectrum_folder_path, spectrum_reference_name, spectrum_file_type)

    if spectrum_names != image_names:
        print("Not all images have spectrums or vise versa or not in the correct order")

    # Load images
    images, image_reference = load_images(image_paths, image_reference_path)

    # Load spectrums
    spectrums, spectrum_reference = load_spectrums(spectrum_paths, spectrum_reference_path)

    # Load quantum efficiency
    qe_spectrum_blue, qe_spectrum_green, qe_spectrum_red = load_qe_all_colors(qe_paths)

    # Interpolate qe to the spectrum
    qe_interpolated = interpolate_qe(spectrums[0,:,0], qe_spectrum_blue, qe_spectrum_green, qe_spectrum_red)
    
    # Relative reflection image
    RR_images = np.zeros(shape=images.shape, dtype='float64') #(#Images, #Rows, #Columns, #Colors)
    i=0
    for image in images:
        RR_images[i] = hadamard_division(image, image_reference)
        i=i+1

    # Relative reflection spectrum
    RR_spectrums = np.zeros(shape=(len(spectrums), len(spectrums[0])), dtype=spectrums.dtype)

    # Dividing y on the reference y
    RR_spectrums[:,:] = np.divide(spectrums[:,:,1], spectrum_reference[:,1])
    
    # We will separate the x_axis from now: 
    x_lambda = spectrums[0,:,0]

    # Find the spectrum that the pixels register
    RR_qe_shape = RR_spectrums.shape + (3,) # (#spectrums, length of spectrum, # colors, first dim is original)
    RR_qe = np.zeros(shape=RR_qe_shape, dtype=RR_spectrums.dtype)
    RR_qe_minus_one = np.array(RR_qe)

    for i in range(len(RR_spectrums)):
        for j in range(3):
            RR_qe[i,:,j]           = np.multiply(RR_spectrums[i,:], qe_interpolated[:,j])
            RR_qe_minus_one[i,:,j] = np.multiply(RR_spectrums[i,:]-1, qe_interpolated[:,j])

    # Spatial average across the image
    spectral_average = np.average(RR_qe, axis=1)

    # Spectral average along all wavelengths
    spatial_average = np.average(RR_images, axis=(1,2))
    
    # To compare the spatial average with the spectral average we divide on upon the other
    comparison = hadamard_division(spatial_average, spectral_average)
    
    # Visualization
    #plot_bgr("Test", RR_qe[0], x_lambda)
    title = "Spatial average divided by spectral average"
    plot_bgr(title, comparison, spectrum_names)
    pyplot.show()


    # Hadamard division
    RR_negatives = np.zeros(shape=images.shape, dtype=images.dtype)
    RR_positives = np.zeros(shape=images.shape, dtype=images.dtype)

    i = 0
    for image in images:
        RR_negatives[i], RR_positives[i] = hadamard_two_face(image, image_reference, image_dark_limit)
        i=i+1



if __name__ == "__main__":
    main()