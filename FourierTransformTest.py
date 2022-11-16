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

def main(): 
    image = plt.imread("huntertestcar.jpg")
    image = image[:, :, :3].mean(axis=2)
    print(image.shape)
    #plt.set_cmap("gray")
    #plt.imshow(image)
    #plt.axis("off")
    #plt.show()

    transformed_Image = np.fft.fftshift(np.fft.fft2(image))
    transformed_Image = np.log(abs(transformed_Image))
    transformed_Image_Plot = plt.plot(transformed_Image)
    #fig = plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(transformed_Image, cmap='gray');
    plt.show()

    img = fig2img(fig)

    img.show()


if __name__ =="__main__":
    main()







