import sys

from hibachi.hib import Run

def run_hibachi():
    try:
        Run()
    except KeyboardInterrupt:
        sys.exit(1)
