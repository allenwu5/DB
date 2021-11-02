import argparse
import csv
from os import listdir
from os.path import exists, join, splitext
from pathlib import Path

import cv2
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm


class Prediction():
    def __init__(self, polygon, conf):
        self.polygon = polygon
        self.conf = conf


def main(opt):
    Path(opt.crop_dir).mkdir(parents=True, exist_ok=True)

    file_to_bboxs = [['name', 'x1', 'y1', 'x2', 'y2']]
    for input_text_name in tqdm(listdir(opt.prediction_dir)):
        name, ext = splitext(input_text_name)
        name = name[4:]
        if ext == '.txt':
            file_to_bbox = [name]
            text_path = join(opt.prediction_dir, input_text_name)
            img_path = join(opt.img_dir, f'{name}.jpg')
            assert exists(img_path), f'{img_path} does not exist'

            predictions = []
            with open(text_path, 'r') as f:
                lines = f.readlines()

                for l in lines:
                    # Last number is confidence
                    polygon_and_conf = l.split(',')
                    conf = float(polygon_and_conf[-1])
                    polygon_coords = [int(x) for x in polygon_and_conf[:-1]]
                    polygon_coords = zip(
                        polygon_coords[::2], polygon_coords[1::2])
                    p = Polygon(polygon_coords)
                    predictions.append(Prediction(p, conf))

            if predictions:
                # sort by confidence
                predictions = sorted(
                    predictions, key=lambda p: p.conf, reverse=True)

                main_polygon = predictions[0].polygon
                main_conf = predictions[0].conf
                region = scale(main_polygon, xfact=5, yfact=1.5)

                for p in predictions[1:]:
                    in_region = region.intersects(p.polygon)
                    if p.conf > 0.45 and in_region:
                        main_polygon = unary_union([p.polygon, main_polygon])

                main_polygon = [int(x) for x in main_polygon.bounds]
                img = cv2.imread(img_path)
                cropped = img[main_polygon[1]:main_polygon[3],
                              main_polygon[0]:main_polygon[2]]
                h, w, c = cropped.shape
                if w > 0 and h > 0:
                    output_path = join(opt.crop_dir, f'{name}.jpg')
                    cv2.imwrite(output_path, cropped)
                    file_to_bbox += main_polygon
            else:
                print(f'{img_path} has no polygon')

            file_to_bboxs.append(file_to_bbox)

    with open(opt.crop_csv_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(file_to_bboxs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='')
    parser.add_argument('--prediction_dir', help='')
    parser.add_argument('--crop_dir', help='')
    parser.add_argument('--crop_csv_path', help='')
    opt = parser.parse_args()
    main(opt)
