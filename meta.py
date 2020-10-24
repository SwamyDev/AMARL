import sys


if __name__ == "__main__":
    with open('amarl/_version.py', 'w') as f:
        f.write(f"__version__ = '{sys.argv[1]}'\n")

