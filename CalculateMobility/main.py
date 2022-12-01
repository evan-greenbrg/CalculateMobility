import argparse
import platform
from multiprocessing import set_start_method

from puller import get_paths
from puller import get_paths_dswe
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
    if args.metrics == 'dswe':
        paths = get_paths_dswe(args.out)
    else:
        paths = get_paths(args.out)

    print('Pulling Mobility')
    rivers = get_mobility_rivers(args.poly, paths, args.river)

    print('Making Gif')
    if (args.metrics == 'single'):
        dswe = False
    elif (args.metrics== 'dswe'):
        dswe = True
    make_gifs(args.river, args.out, dswe=dswe)
