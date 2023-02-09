import pytesseract
import cv2
from PIL import Image

image = cv2.imread("ic/Fila v4.jpg")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("temp/gray.jpg", gray)

blur = cv2.GaussianBlur(gray, (7,7),0)
cv2.imwrite("temp/blur.jpg", blur)

thresh, im_bw = cv2.threshold(gray, 150, 250, cv2.THRESH_BINARY)
cv2.imwrite("temp/thresh.jpg", thresh)
cv2.imwrite("temp/bw.jpg", im_bw)


kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
cv2.imwrite("temp/kernal.png",kernal)

dilate = cv2.dilate(thresh, kernal, iterations=1)
cv2.imwrite("temp/dilate.png",dilate)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=10)
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=10)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return(image)

no_noise = noise_removal(gray)
cv2.imwrite("temp/no_noise.jpg", no_noise)

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return(image)

eroded_image = thin_font(no_noise)
cv2.imwrite("temp/eroded_image.jpg", eroded_image)


def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=10)
    image = cv2.bitwise_not(image)
    return(image)

dilated_image = thin_font(no_noise)
cv2.imwrite("temp/dilated_image.jpg", dilated_image)

inverted_image = cv2.bitwise_not(dilated_image)
cv2.imwrite("temp/inverted.jpg", inverted_image)

dilated_image = "temp/dilated_image.jpg"
inverted_image = "temp\inverted.jpg"



image1 = Image.open(dilated_image)
ocr_result = pytesseract.image_to_string(image1)
print(ocr_result)
    






