#!/usr/bin/env python

"""
Collect results of experiments.

Example:
    collect_results.py n_hidden=100 exp_mb_size=1000
"""

import cPickle, os, sys


def main():
    # Parse arguments.
    constraints = {}
    for arg in sys.argv[1:]:
        k, v = arg.split('=')
        constraints[k] = float(v)
    # Latest Git repo revision where default options were updated.
    latest_version = '3d5f20b'
    # Map an option to its default value, depending on the Git repo revision.
    defaults = {
            '3d5f20b': {
                }
            }
    default = defaults[latest_version]
    # Find all experiments.
    all_states = []
    for root, dirs, files in os.walk('.'):
        for f_name in files:
            if f_name == 'state.pkl':
                # Load experiment settings.
                f_path = os.path.join(root, f_name)
                f_in = open(f_path, 'rb')
                state = cPickle.load(f_in)
                f_in.close()
                # Fill default values.
                for k, v in default.iteritems():
                    if k not in state:
                        state[k] = v
                ok = True
                for k, v in constraints.iteritems():
                    if state[k] != v:
                        ok = False
                        break
                if not ok:
                    # This experiment does not meet the user-specified
                    # constraints.
                    continue
                state['__expdir__'] = root
                all_states.append(state)

    print [s['__expdir__'] for s in all_states]

    return 0


if __name__ == '__main__':
    sys.exit(main())

