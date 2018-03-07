import math
import numpy as np

from ParsingProcesser import ParsingProcesser

FRACTION = 0.1

class Splitter:
    def __init__(self, data_df):
        self.data_df = data_df
    
    def divide_data_random(self, training_fraction):
        if training_fraction < 0.0 or training_fraction > 1.0:
            raise Exception("Invalid training fraction")
        total_num_items = len(self.data_df)
        num_fraction_items = math.floor( training_fraction * total_num_items )
        arr = np.random.random_integers(0, total_num_items - 1, num_fraction_items)
        print(arr)


if __name__ == '__main__':
    parser = ParsingProcesser()
    df, categories_map = parser.build_dataframe()
    splitter = Splitter(df)