import joblib
import os
import sys
import multiprocessing
from glob import glob
import subprocess
import contextlib
import argparse
# import prox.main
# from prox.cmd_parser import parse_config

NUM_THREADS = 3


# @contextlib.contextmanager
# def redirect_argv(l_args):
#     sys._argv = sys.argv[:]
#     sys.argv = l_args
#     yield
#     sys.argv = sys._argv


def worker(thread_no, my_recordings):
    print('Worker:', thread_no)

    for rec in my_recordings:
        # with redirect_argv(['--config=../cfg_files/SLP_novposer.yaml', '--recording_dir={}'.format(rec)]):
        #     print(sys.argv)
        #     args = parse_config()
        #     prox.main.main(**args)

        cmd = 'python prox/main.py --config cfg_files/SLP_novposer.yaml --recording_dir={}'.format(rec[:-1])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    args = parser.parse_args()

    thread_id = int(args.id)

    all_recordings = glob('slp_tform/recordings/*/')
    all_recordings.sort()

    # jobs = []
    # for i in range(NUM_THREADS):
    #     my_recordings = [all_recordings[i] for i in range(i, len(all_recordings), NUM_THREADS)]
    #     p = multiprocessing.Process(target=worker, args=(i, my_recordings))
    #     jobs.append(p)
    #     p.start()

    my_recordings = [all_recordings[i] for i in range(thread_id, len(all_recordings), NUM_THREADS)]
    worker(thread_id, my_recordings)
