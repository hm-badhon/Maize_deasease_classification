import os
def get_file_path(filename):
    """
    Returns the absolute path of the given filename.
    
    :param filename: Name of the file
    :return: Absolute path to the file
    """
    return os.path.abspath(filename)

if __name__ == "__main__":
    # Example usage
    file_name = "example.txt"
    print(f"The absolute path of '{file_name}' is: {get_file_path(file_name)}")