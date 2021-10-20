from os import listdir
from os.path import join, splitext, exists
from pathlib import Path

import cv2
from shapely.geometry import Polygon
from shapely.affinity import scale
from tqdm import tqdm

input_txt_dir = '/content/drive/MyDrive/colab/output'
input_img_dir = '/content/drive/MyDrive/tradevan/2021_china_steel_ocr/public_testing_data'

output_dir = '/content/drive/MyDrive/colab/cropped'
Path(output_dir).mkdir(parents=True, exist_ok=True)

xfact=1.2
yfact=1.8

# 6PWTVzG5LII6tfMyj5JdtHbmaOgOgO.jpg
# res_6PWTVzG5LII6tfMyj5JdtHbmaOgOgO.txt
for input_text_name in tqdm(listdir(input_txt_dir)):
    name, ext = splitext(input_text_name)
    name = name[4:]
    if ext == '.txt':
        text_path = join(input_txt_dir, input_text_name)
        img_path = join(input_img_dir, f'{name}.jpg')
        assert exists(img_path), f'{img_path} does not exist'

        max_conf = 0
        max_conf_bbox = None
        with open(text_path, 'r') as f:
            lines = f.readlines()

            for l in lines:
                # Last one is confidence
                polygon_and_conf = l.split(',')
                conf = float(polygon_and_conf[-1])
                polygon_coords = [int(x) for x in polygon_and_conf[:-1]]
                polygon_coords = zip(polygon_coords[::2], polygon_coords[1::2])
                p = Polygon(polygon_coords)
                p = scale(p, xfact=xfact, yfact=yfact)
                bbox = p.bounds
                if conf > max_conf:
                    max_conf = conf
                    max_conf_bbox = [int(x) for x in bbox]

        if max_conf_bbox is not None:
            img = cv2.imread(img_path)

            cropped = img[max_conf_bbox[1]:max_conf_bbox[3],
                        max_conf_bbox[0]:max_conf_bbox[2]]
            output_path = join(output_dir, f'{name}.jpg')
            cv2.imwrite(output_path, cropped)
        else:
            print(f'{img_path} has no polygon')
