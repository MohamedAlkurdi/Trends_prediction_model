import os

def print_directory_tree(folder_path, exclude_folders, indent=''):
    for item in os.listdir(folder_path):
        if item in exclude_folders:
            continue  # Skip excluded folders

        item_path = os.path.join(folder_path, item)
        print(indent + '├── ' + item)
        if os.path.isdir(item_path):
            print_directory_tree(item_path, exclude_folders, indent + '│   ')

if __name__ == "__main__":
    target_folder ="C:/Users/alkrd/Desktop/graduation_project/the_project"
    exclude_folders = ['venv']
    
    if os.path.isdir(target_folder):
        print(target_folder)
        print_directory_tree(target_folder, exclude_folders)
    else:
        print(f"Error: '{target_folder}' is not a valid directory.")