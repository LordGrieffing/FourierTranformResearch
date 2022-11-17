import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)

    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))

def bandPassFilter(inputArray, filterStart, filterEnd):
    xLength, yLength =inputArray.shape
    blankArray = np.zeros(inputArray.shape)
    workArray = inputArray
    for i in range(xLength):
        for j in range(yLength):
            complexNum = inputArray[i][j]
            realNum = complexNum.real

            if realNum < filterStart or realNum > filterEnd:
                blankArray[i][j] = 1


    for i in range(xLength):
        for j in range(yLength):
            if blankArray[i][j] == 1:
                workArray[i][j] = workArray[i][j] - workArray[i][j] 
    
    return workArray

def zeroCheck(Array):
    xLength, yLength = Array.shape
    for i in range(xLength):
        for j in range(yLength):
            if Array[i][j] == 0j:
                print("Found Zero")
                pass
    pass

def mainTwo():
    #reading the image
    image = plt.imread("images/LadySitting.jpg")
    image = image[:, :, :3].mean(axis=2)
    print(image.shape)

    transformed_Image = calculate_2dft(image)
    transformed_Image_Freq = np.fft.fftfreq(transformed_Image)
    #plt.plot(transformed_Image_Freq, (transformed_Image.real**2) + transformed_Image.imag**2)
    #transformed_Image_display = np.log(abs(transformed_Image))

    filtered_Fourier = bandPassFilter(transformed_Image, 4, 500)
    filtered_Fourier_Freq = np.fft.fftfreq(filtered_Fourier)
    #filtered_Fourier_display = np.log(abs(filtered_Fourier))
    
    plt.subplot(121)
    plt.imshow(transformed_Image_Freq, (transformed_Image.real**2) + transformed_Image.imag**2)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(filtered_Fourier_Freq, (filtered_Fourier.real**2) + filtered_Fourier.imag**2)
    plt.axis("off")
    plt.show()



def main(): 

    #reading the image
    image = plt.imread("images/LadySitting.jpg")
    image = image[:, :, :3].mean(axis=2)
    print(image.shape)

    #array dimmensions
    array_size = len(image)
    centre = int((array_size - 1) / 2)

    coords_left_half = ((x, y) for x in range(array_size) for y in range(centre+1))

    coords_left_half = sorted(coords_left_half,key=lambda x: calculate_distance_from_centre(x, centre))


    transformed_Image = calculate_2dft(image)
    transformed_Image_display = np.log(abs(transformed_Image))
    plt.plot(transformed_Image_display)
    plt.figure()
    plt.imshow(transformed_Image_display, cmap='gray')
    plt.show()
    plt.pause(2)

    #reconstruct the image
    fig = plt.figure()

    rec_image = np.zeros(image.shape)
    individual_grating = np.zeros(image.shape, dtype="complex")

    idx = 0

    #We are doing the filtering here
    #filtered_Fourier = bandPassFilter(transformed_Image, 4, 500)
    #print(filtered_Fourier)
    #plt.plot(filtered_Fourier)
    #plt.imshow(filtered_Fourier, cmap="gray")
    #plt.show()

    display_all_until = 20
    display_step = 10
    next_display = display_all_until + display_step

    for coords in coords_left_half:
        if not (coords[1] == centre and coords[0] > centre):
            idx += 1
        symm_coords = find_symmetric_coordinates(
            coords, centre
        )
        # Copy values from Fourier transform into
        # individual_grating for the pair of points in
        # current iteration
        individual_grating[coords] = transformed_Image[coords]
        individual_grating[symm_coords] = transformed_Image[symm_coords]
        # Calculate inverse Fourier transform to give the
        # reconstructed grating. Add this reconstructed
        # grating to the reconstructed image
        rec_grating = calculate_2dift(individual_grating)
        rec_image += rec_grating
        # Clear individual_grating array, ready for
        # next iteration
        individual_grating[coords] = 0
        individual_grating[symm_coords] = 0
        # Don't display every step
        if idx < display_all_until or idx == next_display:
            if idx > display_all_until:
                next_display += display_step
                # Accelerate animation the further the
                # iteration runs by increasing
                # display_step
                display_step += 100
            display_plots(rec_grating, rec_image, idx)
    plt.show()

    


if __name__ =="__main__":
    main()
    #mainTwo()







