import csv
from os.path import join

from tqdm import tqdm

label_csv = 'G:/我的雲端硬碟/tradevan/2021_china_steel_ocr/標記與資料說明/public_training_data.csv'
train_image_dir = 'G:/我的雲端硬碟/colab/DB/datasets/china_steel/train_images'
train_gts_dir = 'G:/我的雲端硬碟/colab/DB/datasets/china_steel/train_gts'

ext = '.jpg'
filenames = []

with open(label_csv, 'r') as f:
    reader = csv.DictReader(f)
    for d in tqdm(reader):
        filename = d['filename']
        filenames.append(filename)
        label = d['label']
        poly = []
        # Poly
        # E.g. 334,842,569,804,599,983,364,1021,0
        poly += [d['top left x'], d['top left y']]
        poly += [d['top right x'], d['top right y']]
        poly += [d['bottom right x'], d['bottom right y']]
        poly += [d['bottom left x'], d['bottom left y']]
        poly = [str(int(float(x))) for x in poly]

        poly_class = '0'
        bbox_label = poly + [poly_class]
        label_file = join(train_gts_dir, f'{filename}{ext}.txt')
        with open(label_file, 'w') as l:
            l.write(','.join(bbox_label))

image_list_path = 'G:/我的雲端硬碟/colab/DB/datasets/china_steel/train_list.txt'
with open(image_list_path, 'w') as f:
    for filename in filenames:
        f.write(f'{filename}{ext}\n')
