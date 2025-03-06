import os
import re

def delete_files():
    # Loop through all files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.pkl'):
            os.remove(filename)

if __name__ == "__main__":
    delete_files()

