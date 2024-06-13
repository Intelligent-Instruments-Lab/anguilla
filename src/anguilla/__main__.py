import sys

from iipyper import run

from anguilla.app import *

def help():
    print("""
    available subcommands:
        server: run the anguilla OSC server
    """)

def _main():
    # print(sys.argv)
    try:
        if sys.argv[1] == 'server':
            sys.argv = sys.argv[1:]
            run(server)
        else:
            help()
    except IndexError:
        help()

if __name__=='__main__':
    _main()