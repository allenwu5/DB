import argparse
import csv
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm
from os.path import join, exists
import cv2 as cv
from pathlib import Path
import numpy as np

isClosed = True
thickness = 2
show_hard_samples_limit = 100

def main(opt):
    Path(opt.output_dir).mkdir(parents=True, exist_ok=True)

    gts = {}
    with open(opt.gt_rect_csv_path) as f:
        result = csv.DictReader(f)
        for d in result:
            file_name = d['filename']
            # label = d['label']
            poly = []
            poly += [d['top left x'], d['top left y']]
            poly += [d['top right x'], d['top right y']]
            poly += [d['bottom right x'], d['bottom right y']]
            poly += [d['bottom left x'], d['bottom left y']]
            poly = [float(x) for x in poly]
            polygon_coords = zip(poly[::2], poly[1::2])
            p = Polygon(polygon_coords)
            gts[file_name]=p

    ious = {}
    pred = {}
    with open(opt.rect_csv_path) as f:
        result = csv.DictReader(f)
        for d in tqdm(result):
            file_name = d['name']
            poly = []
            poly += [d['x1'], d['y1']]
            poly += [d['x2'], d['y1']]
            poly += [d['x2'], d['y2']]
            poly += [d['x1'], d['y2']]
            poly = [float(x) for x in poly]
            polygon_coords = zip(poly[::2], poly[1::2])
            p = Polygon(polygon_coords)
            pred[file_name]=p

            gt = gts[file_name]
            union = unary_union([p, gt])
            iou = p.intersection(gt).area/union.area
            ious[file_name]=iou

    print(f'avg. IOU: {sum(ious.values())/len(ious)}')            
            
    ious = {k: v for k, v in sorted(ious.items(), key=lambda item: item[1])}            
    for name in list(ious.keys())[:show_hard_samples_limit]:
        img_path=join(opt.img_dir, f'{name}.jpg')
        assert exists(img_path), f'{img_path} does NOT exist!'
        img=cv.imread(img_path)

        iou=ious[name]
        output_path=join(opt.output_dir, f'{iou:2f} {name}.jpg')
        
        for polygon, color in zip([pred[name], gts[name]], [(255, 0, 0), (0, 255, 0)]):
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exterior = [int_coords(polygon.exterior.coords)]
            cv.polylines(img, exterior, isClosed, color, thickness)

        cv.imwrite(output_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='')
    parser.add_argument('--rect_csv_path', help='')
    parser.add_argument('--gt_rect_csv_path', help='')
    parser.add_argument('--output_dir', help='')
    opt = parser.parse_args()
    main(opt)