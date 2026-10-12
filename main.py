import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

from utils.parseargs import parse_argu
from transferlearn.runtransferlearn import run_generate_target_model
from attack.runattack import run_attack_target_model


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _run_experiment(args, log_path):
    print(f'[INFO] Log file: {log_path}')

    if args.exp_mode == 'generate_target_model':
        print('Generating target model...')
        run_generate_target_model(args)

    elif args.exp_mode == 'attack_target_model':
        print('Attacking target model...')
        run_attack_target_model(args)

def main():
    args = parse_argu()
    log_path = os.path.join(args.save_path, 'log')

    with open(log_path, 'a', buffering=1) as log_file:
        tee_out = _Tee(sys.__stdout__, log_file)
        tee_err = _Tee(sys.__stderr__, log_file)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            try:
                _run_experiment(args, log_path)
            except Exception:
                traceback.print_exc()
                raise
            finally:
                tee_out.flush()
                tee_err.flush()


if __name__ == "__main__":
    main()
