import os
import zipfile

def zip_python_files(zip_filename="python_files.zip", start_dir=".", script_filename="zip.py"):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(start_dir):
            # Skip the .venv directory
            if '.venv' in subfolders:
                subfolders.remove('.venv')

            for filename in filenames:
                if filename.endswith('.py'):
                    # Skip the script file itself and the zip file if they appear in the folder
                    if filename == script_filename or filename == os.path.basename(zip_filename):
                        continue

                    filepath = os.path.join(foldername, filename)
                    arcname = os.path.relpath(filepath, start_dir)
                    zipf.write(filepath, arcname)
    print(f"Zipped all Python files excluding .venv, {script_filename}, and {zip_filename} into {zip_filename}")

if __name__ == "__main__":
    zip_python_files()
