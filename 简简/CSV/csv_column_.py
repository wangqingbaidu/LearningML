import pandas as pd


def csv_column_modify (input_file, spec_column, from_content, to_content, save_to):
    data = pd.read_csv(input_file)
    for i in data[spec_column].index:
        if data[spec_column][i] == from_content:
            data[spec_column][i] = to_content

    data.to_csv(save_to, index=False, encoding='utf_8_sig')


spec_column: str = 'other'
from_content, to_content = 5, 4
input_file = 'C:/Users/ysl-pc/Desktop/简简/CSV文件读取/csv_test.csv'
save_to: str = 'C:/Users/ysl-pc/Desktop/简简/CSV文件读取/new_csv_test.csv'
csv_column_modify(input_file, spec_column, from_content, to_content, save_to)
