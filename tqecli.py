#!/usr/bin/env python

import argparse
import logging

from tqe.utils import colors

from tqe import setupSubparsers, run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Train Translation Quality Estimation')
    parser.add_argument('--no-colors', action='store_true',
                        help='Don\'t show colors in verbose log')

    setupSubparsers(parser)

    args = parser.parse_args()

    if args.no_colors:
        colors.disable()

    logLevel = logging.NOTSET
    logging.basicConfig(
        level=logLevel,
        format=(colors.enclose('%(asctime)s', colors.CYAN) +
                colors.enclose('.%(msecs)03d ', colors.BLUE) +
                colors.enclose('%(name)s', colors.YELLOW) + ': ' +
                '%(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    run(args)
