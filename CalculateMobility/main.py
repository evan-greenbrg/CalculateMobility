import argparse
import platform
from multiprocessing import set_start_method

from puller import get_paths
from mobility import get_mobility_rivers
from gif import make_gifs


if __name__ == '__main__':
    if platform.system() == "Darwin":
        set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Pull Mobility')
    parser.add_argument('--poly', metavar='poly', type=str,
                        help='In path for the geopackage path')

    parser.add_argument('--metrics', metavar='metrics', type=str,
                        choices=['single', 'dswe', 'false'],
                        help='Do you want to make the gif?')

    parser.add_argument('--out', metavar='out', type=str,
                        help='output root directory')

    parser.add_argument('--river', metavar='r', type=str,
                        help='River name')

    args = parser.parse_args()

    export_images = False
    paths = get_paths(args.poly, args.out)

    print('Pulling Mobility')
    rivers = get_mobility_rivers(args.poly, paths, args.out, args.river)

    if (args.metrics== 'single'):
        print('Making Gif')
        make_gifs(args.river, args.out)

    elif (args.metrics== 'dswe'):
        print('Making Gif')
        make_gifs_dswe(args.river, args.out)

