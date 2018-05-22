"""
Originally from the CS230 code examples repo - https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/build_dataset.py
The script was modified for the purposes of this project

Split the dataset into train/val/test and resize images to 64x64.

The iMaterialist - furniture dataset comes into the following format:
    data/train/
        .jpeg
        ...
    data/test/
        .jpeg
        ...

Original images have varying sizes.
Resizing to (64, 64) reduces the dataset size, and loading smaller imagesmakes training faster.

We already have a test set created, so we only need to split "train" into train and val sets.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/', help="Directory with the SIGNS dataset")
parser.add_argument('--outputDir', default='../data64x64', help="Where to write the new data")


def resizeAndSave(filename, outputDir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `outputDir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(outputDir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    # trainDataDir = os.path.join(args.data_dir, 'train')
    valDataDir = os.path.join(args.data_dir, 'valid')
    # testDataDir = os.path.join(args.data_dir, 'test')

    # Get the filenames in each directory (train and test)
    # trainFilenames = os.listdir(trainDataDir)
    # trainFilenames = [os.path.join(trainDataDir, f) for f in filenames if f.endswith('.jpeg')]

    print('valDataDir' + valDataDir)
    valFilenames = os.listdir(valDataDir)
    valFilenames = [os.path.join(valDataDir, f) for f in valFilenames if f.endswith('.jpeg')]
    print(valFilenames)

    # testFilenames = os.listdir(testDataDir)
    # testFilenames = [os.path.join(testDataDir, f) for f in testFilenames if f.endswith('.jpeg')]

    filenames = {
      # 'train': trainFilenames,
      'val': valFilenames,
      # 'test': testFilenames
    }

    if not os.path.exists(args.outputDir):
        os.mkdir(args.outputDir)
    else:
        print("Warning: output dir {} already exists".format(args.outputDir))

    # Preprocess train, val and test
    # for split in ['train', 'val', 'test']:
    for split in ['val']:
        output_dir_split = os.path.join(args.outputDir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resizeAndSave(filename, output_dir_split, size=SIZE)

    print("Done building dataset")