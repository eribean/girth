import json
import argparse

import numpy as np
import scipy as sp

import girth

parser_description = """Runs simulations of item response models given a
                        configuration file.  For more information see
                        documentation at

                              https://eribean.github.io/girth/"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="GIRTH Performance Script",
                                     description=parser_description)
    parser.add_argument("--config", help="JSON file of script parameters")
    args = parser.parse_args()

    print(args.config)
