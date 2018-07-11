#!/usr/bin/env python2
# -"- coding: utf-8 -"-


from argparse import ArgumentParser
import os
import re
from subprocess import call
from tgen.logf import log_info

MY_PATH = os.path.dirname(os.path.abspath(__file__))


def lcall(arg_str):
    log_info(arg_str)
    return call(arg_str, shell=True)


def get_confidence(metric, lines):
    for idx, line in enumerate(lines):
        if line.startswith(metric):
            lines = lines[idx:]
            break
    for idx, line in enumerate(lines):
        if line.startswith('Confidence of [Sys1'):
            return line.strip()
    return '???'


def process_all(args):
    txt2sgm = os.path.join(MY_PATH, 'txt2sgm.py')
    gen_log = os.path.join(MY_PATH, 'mteval-v13a-sig.pl')
    bootstrap = os.path.join(MY_PATH, 'paired_bootstrap_resampling_bleu_v13a.pl')

    # create the ref and context SGM files
    lcall("%s -t src -n OST -l any %s %s/context.sgm" %
          (txt2sgm, args.context_fname, args.target_dir))
    lcall("%s -t ref -m -n OST -l en %s %s/ref.sgm" %
          (txt2sgm, args.ref_fname, args.target_dir))

    # create SGM files for all systems
    exp_ids = []
    for exp_dir in args.experiment_dirs:
        exp_id = re.sub(r'/', '_', exp_dir)
        lcall("%s -t test -n OST -l en -s %s %s/%s %s/%s.sgm" %
              (txt2sgm, exp_id, exp_dir, args.output_fname, args.target_dir, exp_id))
        exp_ids.append(exp_id)

    # compute ngram stats for all systems
    os.chdir(args.target_dir)
    for exp_id in exp_ids:
        lcall("%s -s context.sgm -r ref.sgm -t %s.sgm -f %s.log.txt > %s.score.txt" %
              (gen_log, exp_id, exp_id, exp_id))

    # compute bootstrap for all pairs of systems
    for skip, exp_id1 in enumerate(exp_ids):
        for exp_id2 in exp_ids[skip + 1:]:
            # recompute only if not done already (TODO switch for this)
            out_file = 'bootstrap.%s-vs-%s.txt' % (exp_id1, exp_id2)
            if not os.path.isfile(out_file) or os.stat(out_file).st_size == 0:
                lcall("%s %s.log.txt %s.log.txt 100 %f > %s" %
                      (bootstrap, exp_id1, exp_id2, 1. - (0.01 * args.level), out_file))
            with open(out_file) as fh:
                bootstrap_data = fh.readlines()
                print "%s vs. %s: %s" % (exp_id1, exp_id2, bootstrap_data[1].strip())


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-l', '--level', type=float, help='Significance level (95/99)', default=95)
    ap.add_argument('target_dir', type=str, help='Target directory for bootstrap logs')

    ap.add_argument('context_fname', type=str, help='Path to the context TXT file (same one for all)')
    ap.add_argument('ref_fname', type=str, help='Path to the ground-truth/reference TXT file (same one for all)')

    ap.add_argument('output_fname', type=str, help='Name of the system output TXT file in each experiment directory')
    ap.add_argument('experiment_dirs', nargs='+', type=str, help='Experiment directories to use')
    args = ap.parse_args()

    process_all(args)
