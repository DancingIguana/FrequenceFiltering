import numpy as np

def butterworth(distance:float,radius:float,n:int) -> float:
    """
    Butterworth's filter for frequency

    Parameters:
    -----------------------
    distance: distance between the center pixel of the circle and the current pixel
    radius: radius of the circle of the filter
    n: Butterworth's filter order

    Returns:
    ----------------------
    Butterworh's filter value of the pixel
    """

    return 1 / (1+ (radius/distance)**n)


def get_Fuv_from_gray_image(gray_img:np.array) -> tuple[np.array, np.array]:
    """
    Given a gray image, get the discrete Fourier Transform of the image,
    and also return the normalized Fourier Transform image for visualization
    purposes

    Parameters:
    --------------------
    gray_img: the matrix of the image

    Returns:
    -------------------
    Tuple of the FT of the image and the normalized FT
    """

    # Pass to float type
    gray_img_64 = np.float64(gray_img)

    # Fourier Transform in numpy
    Fuv = np.fft.fft2(gray_img_64)
    Fuv = np.fft.fftshift(Fuv)

    # Getting the normalized FT
    Fuv_abs = np.abs(Fuv)
    Fuv_log = 20 * np.log10(Fuv_abs)
    Fuv_norm = np.uint8(255 * Fuv_log / np.max(Fuv_log))

    return Fuv, Fuv_norm

def get_Huv(centers:dict, size:tuple) -> np.array:
    """
    Given a dictionary of filters (with a center and radius) inside the image
    coordinates, get Butterworth's Band Reject Filter

    Parameters:
    -----------------
    center: dictionary of the filters. Each value must have another dictionary
    with the two following values:
        - "center": the center of the filter (x,y)
        - "radius": the radius of the filter
    size: the size of the image to filter (y,x)
    """
    def calculate_distance(a,b):
        return np.linalg.norm(b-a)
    
    # Get the matrix of y coordinates
    y = list(np.arange(-size[0]/2 + 1, size[0]/2+1))*size[1] # Multiply by number of cols
    y = np.array(y)
    y = y.reshape(size[1],size[0]).T
    
    # Get the matrix of x coordinates
    x = list(np.arange(size[1]/2 + 1, -size[1]/2 + 1, -1))*size[0] # Multiply by number of rows
    x = np.array(x)
    x = x.reshape((size[0],size[1]))

    # From the two matrices form a matrix of tuples (xi,yi)
    coordinates = np.stack((x,y),axis = 2)
    
    # Calculate the distances between the pixels and the centers
    # Then from the distances get the butterworth filter values
    distances = np.zeros((len(centers),size[0],size[1]))
    for i,circle in enumerate(centers):
        center = np.array(centers[circle]["center"])
        radius = np.array(centers[circle]["radius"])
        distances_circle = np.apply_along_axis(calculate_distance,2, coordinates, center)
        distances[i,:,:] = np.apply_along_axis(butterworth,1,distances_circle,radius,4)

    Huv = np.prod(distances, axis = 0)
    
    #Huv = 1 - Huv
    return Huv

def frequency_filter(Fuv, Huv):
    """
    Given the Fourier Transform of an image and the Butterworth's Band Reject Filter
    matrices, return the filtered image.

    Parameters:
    ---------------------
    Fuv: the Fourier Transform matrix of the image
    Huv: Butterworth's Band Reject Filter matrix

    Returns:
    --------------------
    Filtered image
    """

    # Convolution theorem for frequency filtering
    Guv = Huv * Fuv

    # Inverse Fourier Transform
    gxy = np.fft.ifft2(Guv)

    # Remove imaginary component by calculating the magnitude 
    # (IFT may return really small values)
    gxy = np.abs(gxy)

    # Return the image in type uint8 for readability
    return np.uint8(gxy)