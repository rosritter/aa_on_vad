import os
import torchaudio
import argparse

def parse_set():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some strings.")

    # Add a string argument
    parser.add_argument(
        '--set',
        type=str,
        default='train-clean-100',
        help='A string argument to be processed'
    )

    # Parse the arguments
    args = parser.parse_args()
    return args.set


def download_libri(current_path, tset):
    torchaudio.datasets.LIBRISPEECH(
                                    root=current_path, 
                                    url=tset, 
                                    download=True,
                                    )
if __name__ == '__main__':
    current_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    tset = parse_set()
    download_libri(current_path, tset)
