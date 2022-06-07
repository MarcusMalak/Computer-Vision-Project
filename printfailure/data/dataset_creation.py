from importlib.resources import path
import re
import cv2
import numpy as np
import random
import os
import pandas as pd
from pathlib import Path


def augmentate(path):
    img = cv2.imread(path, 0)
    # rotate_img = cv2.rotate(rotateCode= 0,src=img)
    # scaled_img = cv2.resize(src=img, dsize =img.shape,fx = 0.5,fy= 0.5)
    flipped_img = cv2.flip(src=img, flipCode=1)
    # cropped_img = img[0:img.shape[1]//2, 0:img.shape[0]//2]
    # rotated_img_clockwise = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    # rotate_img_counterclockwise = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    rows, cols = img.shape
    shift_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    lower_right_shifted_img = cv2.warpAffine(img, shift_matrix, (cols, rows))

    alpha = random.uniform(1, 3)
    contrast_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            contrast_img[y, x] = np.clip(alpha * img[y, x], 0, 255)

    beta = random.randrange(0, 101)
    brightness_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            brightness_img[y, x] = np.clip(img[y, x] + beta, 0, 255)

    print(f"Done img : {path}")

    out_images = {"original": img, "flipped": flipped_img,
                  "lower_right_shift": lower_right_shifted_img, "contrast": contrast_img, "brightness": brightness_img}
    return out_images


def format_img_name(img_name, augmentation_type):
    img_name = img_name.split(".")[0]
    return img_name + "_" + augmentation_type + ".jpg"


def save_images(images, img_name):
    for augmentation_type, img in images.items():
        out_img_name = format_img_name(img_name, augmentation_type)
        cv2.imwrite(out_img_name, img)


if __name__ == "__main__":
    random.seed(235467065458465467456853256)

    cwd = os.getcwd()
    csv_path = cwd + "/printfailure/data/dataset/CV_Images_12/testing/test2/output/assigned_classes.csv"
    img_dir = cwd + "/printfailure/data/dataset/CV_Images_12/testing/test2"
    out_dir = cwd + "/small_dataset/"
    os.makedirs(out_dir)
    os.makedirs(out_dir+"output")
    img_labels = pd.read_csv(csv_path)

    out_labels = pd.DataFrame(columns=img_labels.columns)

    for idx in range(len(img_labels)):
        img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
        images = augmentate(str(img_path))

        save_images(images, out_dir+ str(img_path).split('/')[-1])
        label = img_labels.iloc[idx, 1]
        for augmentation_type, img in images.items():
            out_labels = out_labels.append(
                {"img": format_img_name(str(img_path).split('/')[-1], augmentation_type), "success": label,
                 "failure": img_labels.iloc[idx, 2]}, ignore_index=True)
        break
    out_labels.to_csv(out_dir+"/output/" + "out.csv", index=False)

