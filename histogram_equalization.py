import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_hist_of_img(img):
    img_hist = np.zeros((256,))
    img_size =img.shape[0]*img.shape[1]
    for i in img.flatten():
        img_hist[i] += 1
    return img_hist

def equalize_hist(img):
    image_size=img.shape[0]*img.shape[1]
    img_hist = find_hist_of_img(img) 
    hist_sum = np.cumsum(img_hist)
    n = (hist_sum*255/image_size).astype('int')
    hist_image = img.copy()
    for i in range(256):
        hist_image[img==i] = n[i]
    return hist_image

image = cv2.imread("cat_with_mask.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equalized_img = equalize_hist(image)
plt.imshow(equalized_img)
plt.show()

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     equalized = equalize_hist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
#     cv2.imshow("equalized video", equalized)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cv2.destroyAllWindows()

