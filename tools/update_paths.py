import os
import re
import sys


def add_trailing_one(match):
    """Helper function to append '/1' to kaggle_handle paths without trailing /number."""
    path = match.group(1)
    if not re.search(
        r"/\d+$", path
    ):  # Only add '/1' if the path does not already end with /number
        return f'{path}/1"'
    return f'{path}"'


def update_kaggle_handles_in_file(file_path):
    """Read file, update kaggle_handle lines, and write changes back to file."""
    with open(file_path, "r") as file:
        content = file.read()

    # Updated pattern to match "kaggle_handle" lines that do not end with "/number"
    pattern = r'^(\s*"kaggle_handle": "kaggle://[^"]+)(?<!/[^/0-9])"$'

    # Update kaggle_handle lines
    updated_content = re.sub(
        pattern, add_trailing_one, content, flags=re.MULTILINE
    )

    # Write changes back if content was updated
    if content != updated_content:
        with open(file_path, "w") as file:
            file.write(updated_content)
        print(f"Updated: {file_path}")
    else:
        print(f"No changes needed: {file_path}")


def update_all_files_in_directory(directory_path):
    """Find all files in directory and apply kaggle_handle updates."""
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(
                "presets.py"
            ):  # Only process "presets.py" files
                file_path = os.path.join(root, file_name)
                update_kaggle_handles_in_file(file_path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_kaggle_handles.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        sys.exit(1)

    update_all_files_in_directory(directory_path)


if __name__ == "__main__":
    main()
