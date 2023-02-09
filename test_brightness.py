import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import pytesseract
import pandas as pd

img_orig = cv2.imread('ic/Iqmal v3.jpg')

def base_All(image):
    x = 11
    y = 110
    w = 960
    h = 485
    cv2.rectangle(image, (x,y),(x+w, y+h), (36,255,12),1)
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite('contour.png', cropped_image )
    return image

cv2.imwrite("temp/all.png",base_All(img_orig))
cropped_image = cv2.imread("contour.png")

def resizer(image,width=500):
    # get widht and height
    h,w,c = image.shape
    
    height = int((h/w)* width )
    size = (width,height)
    image = cv2.resize(image,(width,height))
    return image, size

img_re,size = resizer(cropped_image)

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


rgb = cv2.cvtColor(img_re,cv2.COLOR_BGR2RGB)
magic_image = apply_brightness_contrast(rgb,120,0)
magic_image_c1 = apply_brightness_contrast(rgb,0,40)
cv2.imwrite("temp/bright_image.png",magic_image_c1)
magic_image_c2 = apply_brightness_contrast(rgb,50,40)

#plt.figure(figsize=(15,10))
#plt.subplot(2,2,1)
#plt.imshow(rgb)
#plt.title('Orignal')

#plt.subplot(2,2,2)
#plt.imshow(magic_image)
#plt.title('magic :B = 120')

#plt.subplot(2,2,3)
#plt.imshow(magic_image_c1)
#plt.title('magic :C = 40')


#plt.subplot(2,2,4)
#plt.imshow(magic_image_c2)
#plt.title('magic :B= 50, C = 40')

plt.show()


final_image = cv2.imread("contour.png")
data = pytesseract.image_to_data(final_image)
dataList = list(map(lambda x: x.split('\t'),data.split('\n')))
df = pd.DataFrame(dataList[1:],columns=dataList[0])


image2 = final_image.copy()

level = 'line'
for l,x,y,w,h,c,txt in df[['level','left','top','width','height','conf','text']].values:
    #print(l,x,y,w,h,c)
    if level == 'page':
        if l == 1:
            cv2.rectangle(image2,(x,y),(x+w,y+h),(0,0,0),1)
        else:
            continue
            
    elif level == 'block':
        if l == 2:
            cv2.rectangle(image2,(x,y),(x+w,y+h),(255,0,0),1)
        else:
            continue
    
    elif level == 'para':
        if l == 3:
            cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),1)
        else:
            continue
    
    elif level == 'line':
        if l == 4:
            cv2.rectangle(image2,(x,y),(x+w,y+h),(0,0,255),1)
        else:
            continue
            
    elif level == 'word':
        if l == 5:
            cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.putText(image2,txt,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
        else:
            continue
            
cv2.imshow("bounding box",image2)
cv2.waitKey()
cv2.destroyAllWindows()




