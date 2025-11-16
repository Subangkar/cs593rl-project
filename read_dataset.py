import csv 
import pandas as pd

class DatasetProcessor:
    """
    A class to process dataset files for image captioning tasks.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self.read_dataset()

    def read_dataset(self) -> pd.DataFrame:
        """
        Reads a CSV dataset file and returns a pandas DataFrame.
        Assumes the CSV has columns: 'image_path' and 'caption'.
        """
        data = pd.read_csv(self.file_path)
        return data


    def generate_reward_testing_samples(self, n: int = 10) -> tuple[list[str], list[str]]:
        """
        Generates a set of samples for reward testing.
        Selects the first n entries from the dataset.
        """
        random_df = self.df.sample(n=n, random_state=42)

        # get the 'question' column as a list
        questions = random_df['question'].tolist()
        instructions = random_df['instruction'].tolist()

        return questions, instructions