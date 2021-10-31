import csv
from os.path import join
import argparse
from tqdm import tqdm
import shutil

label_csv = '/content/drive/MyDrive/tradevan/2021_china_steel_ocr/Training Label/public_training_data.csv'
source_image_dir = '/content/public_training_data/public_training_data'
ext = '.jpg'

def create_gts(dicts, gts_dir, dest_image_dir, list_path):
    filenames = []
    for d in tqdm(dicts):
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
        label_file = join(gts_dir, f'{filename}{ext}.txt')
        with open(label_file, 'w') as l:
            l.write(','.join(bbox_label))

        source_image_path = join(source_image_dir, f'{filename}{ext}')
        dest_image_path = join(dest_image_dir, f'{filename}{ext}')
        shutil.move(source_image_path, dest_image_path)

    with open(list_path, 'w') as f:
        for filename in filenames:
            f.write(f'{filename}{ext}\n')

def main(opt):
    dicts = []
    with open(label_csv, 'r') as f:
        for d in csv.DictReader(f):
            dicts.append(d)

    train_dicts = dicts[:opt.train_size]
    val_dicts = dicts[opt.train_size:]

    create_gts(train_dicts, '/content/train_gts', '/content/train_images', '/content/train_list.txt')
    create_gts(val_dicts, '/content/test_gts', '/content/test_images', '/content/test_list.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, help='')
    opt = parser.parse_args()
    main(opt)
