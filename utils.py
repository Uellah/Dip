import os
import numpy as np
import csv




class out_to_file:
    def __init__(self, filename):
        self.filename = filename

        if not os.path.exists('out'):
            os.makedirs('out')

        self.filepath = os.path.join('out', self.filename)

    def clear_file(self):
        with open(self.filepath, 'w') as file:
            pass

    def print_string_to_file(self, content):
        with open(self.filepath, 'a') as file:
            file.write(content + "\n")

    def write_numpy_to_csv(self, array):
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a numpy array")

        with open(self.filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in array:
                writer.writerow(row)
