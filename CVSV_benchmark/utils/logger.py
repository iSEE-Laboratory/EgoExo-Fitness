# encoding: utf-8

import logging
import os
import sys


def setup_logger(name, save_dir, save_name, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    # print logs to screen
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # save logs to file
    if save_dir:
        try:
            fh = logging.FileHandler(os.path.join(save_dir, save_name), mode='w')
        except:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                fh = logging.FileHandler(os.path.join(save_dir, save_name), mode='w')
            else:
                print('*** Wrong when generate logging dir ***')
                exit(-1)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
