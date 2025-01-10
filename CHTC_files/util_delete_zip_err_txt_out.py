import os
import re

def delete_files():
    # Loop through all files in the current directory
    for filename in os.listdir('.'):
        # Check if the file ends with a digit followed by .zip
        if re.search(r'\d\.zip$', filename):
            os.remove(filename)
        # Check if the file ends with .err
        elif filename.endswith('.err') or filename.endswith('.txt') or filename.endswith('.out'):
            os.remove(filename)

if __name__ == "__main__":
    delete_files()