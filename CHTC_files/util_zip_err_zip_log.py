import zipfile
import os
import re

def pack_and_delete_files():
    # Create a ZIP file named mcsolve_result.zip with ZIP64 extensions enabled
    with zipfile.ZipFile('zipped_results.zip', 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        # Loop through all files in the current directory
        for filename in os.listdir('.'):
            # Check if the file is a .err file
            if filename.endswith('.err') or filename.endswith('.log'):
                zipf.write(filename)
                os.remove(filename)  # Delete the file
            # Check if the file is a .zip file ending with an integer
            elif re.search(r'\d\.zip$', filename):
                zipf.write(filename)
                os.remove(filename)  # Delete the file
if __name__ == "__main__":
    pack_and_delete_files()