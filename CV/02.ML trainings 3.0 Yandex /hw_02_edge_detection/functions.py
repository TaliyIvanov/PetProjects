# 1
# do not change the code in the block below
# __________start of block__________
import numpy as np
def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    # First var (by authors)
    # Apply Sobel filter for X and Y gradients using convolution
    # for i in range(1, height + 1):
    #     for j in range(1, width + 1):
    #         region = padded_image[i-1:i+2, j-1:j+2]
    #         gradient_x[i-1, j-1] = np.sum(sobel_x * region)
    #         gradient_y[i-1, j-1] = np.sum(sobel_y * region)

    # Secvond var (by myself)
    # I think my variant more simple
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+3, j:j+3]
            gradient_x[i, j] = np.sum(sobel_x * region)
            gradient_y[i, j] = np.sum(sobel_y * region)
         
    return gradient_x, gradient_y


# 2

import numpy as np # for your convenience when you copy the code to the contest
def compute_gradient_magnitude(sobel_x, sobel_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
    return magnitude

import numpy as np # for your convenience when you copy the code to the contest
def compute_gradient_direction(sobel_x, sobel_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    gradient_direction = np.degrees(np.arctan2(sobel_y, sobel_x))
    return gradient_direction

# 3

import numpy as np

cell_size = 7
def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale
    
    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y) # shape = (28,28)
    direction = compute_gradient_direction(gradient_x, gradient_y) # shape = (28,28)

    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_x = image.shape[1] // cell_width # 28 // 7 = 4
    n_cells_y = image.shape[0] // cell_height # 28 // 7 = 4

    histograms = np.zeros((n_cells_y, n_cells_x, bins)) # shape = (4,4,9)
    bin_width = 180 / bins

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # получаем участок градиентов, соответствующий текущей ячейке
            mag_cell = magnitude[i * cell_height:(i+1) * cell_height,
                                 j * cell_width:(j+1) * cell_width]
            dir_cell = direction[i * cell_height:(i+1) * cell_height,
                                 j * cell_width:(j+1) * cell_width]
            
            # приводим углы к диапазону [0, 180)
            dir_cell = np.mod(dir_cell, 180)

            # теперь двигаемся по пикселям внутри ячейки
            for y in range(cell_height):
                for x in range(cell_width):
                    angle = dir_cell[y, x]
                    mag = mag_cell[y, x]

                    # индекс нижнего и верхнего бина (интерполяция по 2 ближайшим бинам)
                    bin_idx  = int(angle // bin_width)
                    bin_next = (bin_idx + 1) % bins

                    # сколько % отнести в следующий бин
                    ratio = (angle % bin_width) / bin_width
                    histograms[i, j, bin_idx] += mag * (1 - ratio)
                    histograms[i, j, bin_next] += mag * ratio

    return histograms