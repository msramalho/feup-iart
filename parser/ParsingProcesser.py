import pandas as pd
import numpy as np 
import tempfile
import os

class ParsingProcesser:
    '''
    Has the purpose of parsing the data and replacing the category string values to numbers
    '''
    DATASET_PATH = "../dataset/frogs.csv"
    ROW_LABEL_PAIR = (22, 25)
    SEPARATOR = "-"
    INDEX_CATEGORY_LABEL = "category_index"
    TMP_FILENAME = "pandas_tmp_tf_processed"
    FORCE_REPROCESS = False #warning slows down debugging

    def __init__(self, filepath=DATASET_PATH, row_pair = ROW_LABEL_PAIR):
        (row_start, row_past_end) = row_pair
        self.filepath = filepath
        self.start_row = row_start
        self.past_end_row = row_past_end
        self._temp_filepath = self.get_temp_filepath()
        self.df_file = pd.read_csv(self.filepath)

    def build_dataframe(self):
        self._process_data_facade()
        df_data = pd.read_pickle(self._temp_filepath)
        cat_map = self.get_category_map()
        return df_data, cat_map

    def get_temp_filepath(self):
        tmp_dir = tempfile.gettempdir()
        return os.path.join(tmp_dir, self.TMP_FILENAME)

    def get_category_map(self):
        '''
        return a map of type
         Family-Genus-Species: index
         index is to be used by tensorflow
        '''
        return self.categories_map
    

    def _join_row_strings(self, rows, separator = SEPARATOR):
         return separator.join(rows)

    def _build_categories_map(self):
        ''' 
        build an index that converts categories rows to a number for TF
        '''
        _df_parameters, df_categories = self._split_df()
        labels_df = df_categories.drop_duplicates()
        
        categories_map = {}

        i = 0
        for tup in labels_df.itertuples():
            seq = tup[1:]
            catString = self._join_row_strings(seq)
            categories_map[catString] = i
            i = i + 1
        
        self.categories_map = categories_map

    def _convert_categories_to_index(self, df_parameteres, df_categories):
        cat_indexes = []
        for index, rows in df_categories.iterrows():
            seq = rows[0:]
            catString = self._join_row_strings(seq)
            cat_indexes.append(self.categories_map[catString])
        df_cat_indexes = pd.DataFrame(cat_indexes, columns=[self.INDEX_CATEGORY_LABEL])
        tf_df = pd.concat([df_parameteres, df_cat_indexes], axis=1).sort_index()
        return tf_df


    def _split_df(self):
        df_file = self.df_file
        df_parameters = df_file.iloc[ :, :self.start_row ]
        df_categories = df_file.iloc[ :, self.start_row:self.past_end_row]

        return df_parameters, df_categories

    def _process_data(self):
        df_parameters, df_categories = self._split_df()
        self.df_tf_dataset = self._convert_categories_to_index(df_parameters, df_categories)



    def _process_data_facade(self):
        self._build_categories_map()
        bFileExists = os.path.isfile(self._temp_filepath) 
        if ( not bFileExists or self.FORCE_REPROCESS):
            if bFileExists:
                os.remove(self._temp_filepath)
            self._write_to_temp()

    def _write_to_temp(self):
        self._process_data()
        self.df_tf_dataset.to_pickle(self._temp_filepath)



if __name__ == '__main__':
    parser = ParsingProcesser()
    df, cat_map = parser.build_dataframe()
    print(df)