
import os


class Data_injestion:
    def __init__(self):
        pass
    def get_text_from_files(self,root_folder):
        """
        Recursively fetch all .txt files from the root folder, read their content, and return a list of extracted text.
        
        :param root_folder: Path to the root directory
        :return: List of text content from all .txt files
        """
        text_data = []
        for dirpath, _, filenames in os.walk(root_folder):
            for file in filenames:
                if file.endswith(".txt"):
                    file_path = os.path.join(dirpath, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_data.append(f.read())
        return text_data