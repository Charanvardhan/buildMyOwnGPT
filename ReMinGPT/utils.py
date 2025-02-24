"""
utils.py

Utility functions and classes for experiment configuration, logging, and reproducibility in the reminGPT project.

This module provides essential utilities for:
- Setting random seeds to ensure reproducibility.
- Handling configuration management with nested attributes.
- Logging configurations and command-line arguments to maintain experiment records.

Classes:
--------
- CfgNode: A lightweight configuration handler inspired by `yacs`, allowing nested attributes,
           easy overrides from dictionaries and command-line arguments, and serialization to dictionaries.

Functions:
----------
- set_seed(seed): 
    Sets random seeds across Python's `random`, NumPy, and PyTorch libraries to ensure reproducible results.
- setup_logging(config): 
    Creates a working directory, logs command-line arguments (`args.txt`), and saves the current configuration to a JSON file (`config.json`).

Example Usage: At the end of the file.

Generated Files:
----------------
- logs/args.txt: Saves the command-line arguments used to run the script.
- logs/config.json: Saves the experiment configuration in a readable JSON format.

Command-line Override Example:
------------------------------
When running a script that uses this utility, you can pass arguments like:
$ python train.py --model.n_layer=8 --trainer.batch_size=64

This will override the default configuration values with the specified ones.

Safety Checks:
--------------
- Ensures that configuration attributes being overridden exist.
- Uses `literal_eval` to safely parse string values into Python data types (e.g., '42' -> 42, '[1,2,3]' -> [1,2,3]).

Best Practices:
---------------
- Always set a random seed with `set_seed()` for reproducible experiments.
- Use `setup_logging()` at the start of your training script to track configurations and commands.
- Maintain structured configurations using the `CfgNode` class for clarity and ease of use.

"""


import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """ monotonous bookkeeping """
    workDir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(workDir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(workDir, 'args.txt'), 'w') as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(workDir, "config.json"), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """ a ligthweight configuration class inspired by yacs """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)
    
    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent*4)+p for p in parts]
        return "".join(parts)
    
    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k,v in self.__dict__.items() }
    
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    
    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, "expectin each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval

            try:
                val = literal_eval(val)


            except ValueError:
                pass

            
            assert key[:2] == '--'
            key = key[2:]
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            
            leaf_key = keys[-1]

            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)




if __name__ == "__main__":
    config = CfgNode(
        system=CfgNode(work_dir='./logs'),
        model=CfgNode(n_layer=6, hidden_size=256),
        trainer=CfgNode(batch_size=32)
    )

    set_seed(42)
    config.merge_from_args(['--model.n_layer=10', '--trainer.batch_size=64'])
    setup_logging(config)

    print(config)