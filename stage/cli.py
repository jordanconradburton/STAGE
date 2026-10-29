"""
Command-line interface:

    stage train <dataset> [options]               train and save a checkpoint (combines the
                                                  dataset's shards into <dataset>/combined first)
    stage combine <dataset>                       only combine a dataset's shards
    stage evaluate <checkpoint> <test folder>     evaluate on held-out data
    stage sweep <checkpoint> <test folder>...     evaluate on several folders, one summary table

The pipeline and model localize a single drone. Datasets with more than one
drone per experiment are not supported.
"""
import argparse

from . import evaluate, sweep, train
from .data import combine

COMMANDS = {
    "combine": (combine, "Only combine a dataset's per-shard folders into one training folder (train does this itself)."),
    "train": (train, "Train the encoder and position decoder."),
    "evaluate": (evaluate, "Evaluate a checkpoint on a held-out test folder."),
    "sweep": (sweep, "Evaluate a checkpoint on several test folders (e.g. sensor counts) with one summary table."),
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="stage", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, (module, help_text) in COMMANDS.items():
        module.add_arguments(sub.add_parser(name, help=help_text, description=module.__doc__,
                                            formatter_class=argparse.RawDescriptionHelpFormatter))
    args = p.parse_args(argv)
    COMMANDS[args.command][0].run(args)


if __name__ == "__main__":
    main()
