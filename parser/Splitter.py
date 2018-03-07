import math
import numpy as np

from ParsingProcesser import ParsingProcesser

FRACTION = 0.1

class Splitter:
    def __init__(self, data_df):
        self.data_df = data_df
    
    def divide_data_random(self, training_fraction):
        '''
        return 2 pandas dataframe objects
        which are the result of splitting randomly the original data 
        len(training_df) = training_fraction * len(df_data)
        '''
        if training_fraction < 0.0 or training_fraction > 1.0:
            raise Exception("Invalid training fraction")

        total_num_items = len(self.data_df)
        num_fraction_items = math.floor( training_fraction * total_num_items )

        indexes = np.array(range(0, total_num_items))
        training_indexes = np.random.random_integers(0, total_num_items - 1, num_fraction_items)
        test_indexes = np.delete(indexes, training_indexes)

        training_df = self.data_df.ix[training_indexes]
        training_df.sort_index(inplace=True)

        test_df = self.data_df.ix[test_indexes]
        test_df.sort_index(inplace=True)

        return training_df, test_df


if __name__ == '__main__':
    parser = ParsingProcesser()
    df, categories_map = parser.build_dataframe()
    splitter = Splitter(df)
    training, test = splitter.divide_data_random(FRACTION)
    print(training, test)