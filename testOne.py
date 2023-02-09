import cv2
import pytesseract
import numpy as np
import pandas as pd

# Load image, grayscale, Otsu's threshold
image = cv2.imread('ic/Hanis v2.jpg')

def base_All(image):
    x = 15
    y = 120
    w = 960
    h = 485
    cv2.rectangle(image, (x,y),(x+w, y+h), (36,255,12),1)
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite('contour.png', cropped_image )
    return image

cropped_image = cv2.imread("contour.png")
cv2.imwrite("temp/all.png",base_All(image))

gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and remove small noise
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        cv2.drawContours(opening, [c], -1, 0, -1)

# Invert and apply slight Gaussian blur
result = 255 - opening
result = cv2.GaussianBlur(result, (3,3), 0)
cv2.imwrite("temp/result.png",result)

# Perform OCR
data = pytesseract.image_to_string(result, lang='eng', config='--psm 6')
print(data)

#cv2.imshow('thresh', thresh)
#cv2.imshow('opening', opening)
#cv2.imshow('result', result)
#cv2.waitKey()  

final_image = cv2.imread("contour.png")
data = pytesseract.image_to_data(final_image)
dataList = list(map(lambda x: x.split('\t'),data.split('\n')))
df = pd.DataFrame(dataList[1:],columns=dataList[0])
df.head(10)

image2 = final_image.copy()

level = 'block'
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