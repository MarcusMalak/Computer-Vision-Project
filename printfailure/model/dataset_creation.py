import cv2
import numpy as np
import random

def augmentate(path):
    img = cv2.imread(path,0)
    # rotate_img = cv2.rotate(rotateCode= 0,src=img)
    scaled_img = cv2.resize(src=img, dsize =img.shape,fx = 0.5,fy= 0.5)
    flipped_img  = cv2.flip(src=img, flipCode = 1)
    cropped_img = img[0:img.shape[1]//2, 0:img.shape[0]//2]
    rotated_img_clockwise = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    rotate_img_counterclockwise = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    rows,cols = img.shape
    shift_matrix = np.float32([[1,0,100],[0,1,50]])
    lower_right_shifted_img = cv2.warpAffine(img,shift_matrix,(cols,rows))

    alpha = random.uniform(1,3)     
    contrast_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            contrast_img[y,x] = np.clip(alpha*img[y,x], 0, 255)

    beta = random.randrange(0,101)
    brightness_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            brightness_img[y,x] = np.clip(img[y,x] + beta, 0, 255)

    print(f"Done img : {path}")

    out_images= {"scaled":scaled_img,"flipped":flipped_img,"cropped":cropped_img,
    "rotated_clockwise":rotated_img_clockwise,"rotated_counterclockwise":rotate_img_counterclockwise,
    "lower_right_shift":lower_right_shifted_img,"contrast":contrast_img,"brightness":brightness_img}
    return out_images


def save_images(images, img_name):
    for augmentation_type, img in images.items():
        cv2.imwrite(img_name+"_"+augmentation_type+".png",img)



if __name__ == "__main__":
    random.seed(235467065458465467456853256)
    images =augmentate("abc.png")
    save_images(images, "abc")
    

# cv2.imshow("window",img)
# cv2.imwrite("out_img.png",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




