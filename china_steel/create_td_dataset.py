import argparse
import csv
import shutil
from os.path import join
from pathlib import Path

from tqdm import tqdm

img_ext = '.jpg'


def create_dataset(source_img_dir, dataset, gts_dir, dest_img_dir, dest_img_list_path):
    """Create dataset based on https://github.com/MhLiao/DB#prepar-dataset
    """    
    Path(gts_dir).mkdir(parents=True, exist_ok=True)
    Path(dest_img_dir).mkdir(parents=True, exist_ok=True)

    filenames = []
    for d in tqdm(dataset):
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
        poly = [x for x in poly]

        poly_class = '0'
        bbox_label = poly + [poly_class]
        label_file = join(gts_dir, f'{filename}{img_ext}.txt')
        with open(label_file, 'w') as l:
            l.write(','.join(bbox_label))

        source_img_path = join(source_img_dir, f'{filename}{img_ext}')
        dest_img_path = join(dest_img_dir, f'{filename}{img_ext}')
        shutil.move(source_img_path, dest_img_path)

    with open(dest_img_list_path, 'w') as f:
        for filename in filenames:
            f.write(f'{filename}{img_ext}\n')


def main(opt):
    dataset = []
    with open(opt.gt_csv_path, 'r') as f:
        for d in csv.DictReader(f):
            dataset.append(d)

    train_dataset = dataset[:opt.train_size]
    val_dataset = dataset[opt.train_size:]

    create_dataset(opt.img_dir, train_dataset, join(opt.dataset_dir, 'train_gts'), join(
        opt.dataset_dir, 'train_images'), join(opt.dataset_dir, 'train_list.txt'))
    create_dataset(opt.img_dir, val_dataset, join(opt.dataset_dir, 'test_gts'), join(
        opt.dataset_dir, 'test_images'), join(opt.dataset_dir, 'test_list.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv_path', type=str, help='')
    parser.add_argument('--img_dir', type=str, help='')
    parser.add_argument('--dataset_dir', type=str, help='')
    parser.add_argument('--train_size', type=int, help='')
    opt = parser.parse_args()
    main(opt)
