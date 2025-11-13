from codes.utils.r_common import *


class RLogSave:
    def __init__(self):
        self.table_content = [[]]

    def save_log(self, save_path, filename):
        save_dir = os.path.join(save_path, filename + ".csv")
        data_write_csv(save_dir, self.table_content)

    def add_tail(self, row_data):
        self.table_content.append(row_data)

    def set_table_head(self, row_data):
        self.table_content[0] = row_data

    def get_table_rows_num(self):
        return len(self.table_content)-1

    def display_table(self):
        print(self.table_content)

    def clear_all(self):
        self.table_content = [[]]
