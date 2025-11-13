from codes.utils.r_common import *


class RCsvClass:
    def __init__(self):
        self.csv_file = ""
        self.cols_name_list = []
        self.cols_chinese_name_list = []
        self.content = []
        self.rows_count = 0
        self.cols_count = 0

    def open_table(self, dta_path, file_name):
        self.csv_file = os.path.join(dta_path, file_name)
        data_list = data_read_csv(self.csv_file)
        self.cols_name_list = data_list[0]
        data_list.pop(0)
        self.content = data_list
        self.rows_count = len(self.content)
        self.cols_count = len(self.cols_name_list)

    def set_cols_chinese_name(self, cols_chinese_names):
        self.cols_chinese_name_list = cols_chinese_names

    def get_cols_name_list(self):
        return self.cols_name_list.copy()

    def display_table_inf(self):
        print(self.csv_file)
        print("rows_count:", self.rows_count)
        print("cols_count:", self.cols_count)
        print("cols_name:", self.cols_name_list)

    def get_col_idx(self, col_name):
        return self.cols_name_list.index(col_name)

    def save_table(self, save_path, filename):
        save_dir = os.path.join(save_path, filename)
        table_all = [self.cols_name_list] + self.content
        data_write_csv(save_dir, table_all)

    def get_col_datas(self, col_name):
        col_datas = []
        col_idx = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            col_datas.append(self.content[i][col_idx])
        return col_datas

    def get_col_idx_datas(self, col_idx):
        col_datas = []
        # col_idx = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            col_datas.append(self.content[i][col_idx])
        return col_datas

    def get_col_datas_float(self, col_name):
        col_datas = []
        col_idx = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            col_datas.append(float(self.content[i][col_idx]))
        return col_datas

    def get_col_idx_datas_float(self, col_idx):
        col_datas = []
        # col_idx = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            col_datas.append(float(self.content[i][col_idx]))
        return col_datas

    def add_tail(self, row_data):
        self.content.append(row_data)

    def set_table_head(self, table_head):
        self.cols_name_list = table_head
        self.cols_count = len(table_head)

    def set_table_content(self, row_datas_list):
        self.content = row_datas_list
        self.rows_count = len(self.content)

    def get_table_rows_num(self):
        return len(self.content)

    def display_table(self):
        print(self.content)

    def row_to_2d_col(self, row_list):
        col_list = []
        for i in range(len(row_list)):
            col_list.append(row_list[i])
        return col_list

    def clear_all(self):
        self.content = []

    def get_data(self, row, col):
        return self.content[row][col]

    def get_data_by_col_name(self, row, col_name):
        col_idx = self.get_col_idx(col_name)
        return self.content[row][col_idx]

    def set_data(self, row, col, data):
        self.content[row][col] = data

    def set_data_by_col_name(self, row, col_name, data):
        col_idx = self.get_col_idx(col_name)
        self.content[row][col_idx] = data

    def unique_col(self, col_name):
        col_list = []  # self.content[:][self.col_idx(col_name)]
        col_idx = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            col_list.append(self.content[i][col_idx])
        col_unique = list(set(col_list))
        return col_unique

    def unique_col_idx(self, col_idx):
        col_list = []  # self.content[:][self.col_idx(col_name)]
        # col_idx = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            col_list.append(self.content[i][col_idx])
        col_unique = list(set(col_list))
        return col_unique

    def find_data_by_col_name(self, col_name, data):
        idx_col = self.get_col_idx(col_name)
        return self.find_data_by_col_idx(idx_col, data)

    def find_data_by_col_idx(self, col_idx, data):
        idx_list = []
        for i in range(self.rows_count):
            if self.content[i][col_idx] == data:
                idx_list.append(i)
        return idx_list

    def find_data_by_col_name_and_rows_idx_list(self, col_name, rows_idx_list, data):
        idx_list = []
        idx_col = self.get_col_idx(col_name)
        for i in range(len(rows_idx_list)):
            if self.content[rows_idx_list[i]][idx_col] == data:
                idx_list.append(rows_idx_list[i])
        return idx_list

    def find_data_by_col_name_and_datas_list(self, col_name, datas_list):
        idx_list = []
        idx_col = self.get_col_idx(col_name)
        for i in range(self.rows_count):
            if self.content[i][idx_col] in datas_list:
                idx_list.append(i)
        return idx_list

    def find_data_by_col_name_in_datas_list(self, col_name, datas_list):
        idx_list = []
        idx_col = self.get_col_idx(col_name)
        for i in range(len(datas_list)):
            idx_l = self.find_data_by_col_idx(idx_col, datas_list[i])
            if len(idx_l) >= 1:
                idx_list.append(idx_l[0])
            else:
                idx_list.append(-1)
        return idx_list

    def get_datas_by_col_name_and_rows_idx_list(self, col_name, rows_idx_list):
        content = []
        col_idx = self.get_col_idx(col_name)
        for i in range(len(rows_idx_list)):
            content.append(self.content[rows_idx_list[i]][col_idx])
        return content

    def get_datas_by_cols_name_list_and_data(self, cols_name_list, find_col_name, data):
        content = []
        cols_idx_list = []
        find_col_idx = 0
        for i in range(len(cols_name_list)):
            cols_idx_list.append(self.get_col_idx(cols_name_list[i]))
        find_col_idx = self.get_col_idx(find_col_name)
        for i in range(self.rows_count):
            if self.content[i][find_col_idx] == data:
                for j in range(len(cols_idx_list)):
                    content.append(self.content[i][cols_idx_list[j]])
        return content

    def get_datas_by_cols_name_list_and_rows_idx_list(self, cols_name_list, rows_idx_list):
        content = []
        cols_idx_list = []
        for i in range(len(cols_name_list)):
            cols_idx_list.append(self.get_col_idx(cols_name_list[i]))
        if rows_idx_list != "ALL":
            for i in range(len(rows_idx_list)):
                row_datas = []
                for j in range(len(cols_idx_list)):
                    if rows_idx_list[i] >= 0:
                        row_datas.append(self.content[rows_idx_list[i]][cols_idx_list[j]])
                content.append(row_datas)
        else:
            for i in range(self.rows_count):
                row_datas = []
                for j in range(len(cols_idx_list)):
                    row_datas.append(self.content[i][cols_idx_list[j]])
                content.append(row_datas)
        return content

    def get_datas_by_cols_name_list_and_rows_idx_list_1(self, cols_name_list, rows_idx_list):
        content = []
        cols_idx_list = []
        for i in range(len(cols_name_list)):
            cols_idx_list.append(self.get_col_idx(cols_name_list[i]))
        for i in range(len(rows_idx_list)):
            row_datas = []
            for j in range(len(cols_idx_list)):
                if rows_idx_list[i] >= 0:
                    row_datas.append(self.content[rows_idx_list[i]][cols_idx_list[j]])
                else:
                    row_datas.append('nan')
            content.append(row_datas)
        return content

    def set_table_size(self, rows_count, cols_count, cols_name):
        if len(cols_name)==cols_count:
            self.cols_name_list = cols_name
            self.rows_count = rows_count
            self.cols_count = cols_count
            self.content = [["" for _ in range(cols_count)] for _ in range(rows_count)]

    def set_table(self, cols_name, content):
        self.cols_name_list = cols_name
        self.content = content
        self.rows_count = len(content)
        self.cols_count = len(cols_name)


    def set_col_datas(self, col_name, datas_list):
        col_idx = self.get_col_idx(col_name)
        print(len(datas_list))
        for i in range(len(datas_list)):
            self.content[i][col_idx] = datas_list[i]

    def add_col_datas(self, col_name, datas_list):
        if len(datas_list) == self.rows_count:
            self.cols_name_list.append(col_name)
            for i in range(self.rows_count):
                self.content[i].append(datas_list[i])
            self.cols_count += 1
            return True
        else:
            return False

    def add_mul_col_datas(self, cols_name_list, datas_list):
        if len(datas_list) == self.rows_count:
            self.cols_name_list += cols_name_list
            for i in range(self.rows_count):
                self.content[i] += datas_list[i]
            self.cols_count += len(cols_name_list)
            return True
        else:
            return False

    def chang_col_name(self, old_col_name, new_col_name):
        col_idx = self.get_col_idx(old_col_name)
        if col_idx != -1:
            self.cols_name_list[col_idx] = new_col_name
            return True
        return False

    def del_col(self, col_idx):
        self.cols_name_list.pop(col_idx)
        for i in range(self.rows_count):
            self.content[i].pop(col_idx)
        self.cols_count = len(self.cols_name_list)

    def del_cols_by_cols_name_list(self, cols_name_list):
        for j in range(len(cols_name_list)):
            col_idx = self.get_col_idx(cols_name_list[j])
            if col_idx != -1:
                self.del_col(col_idx)

    def flat_list(self, datas_list):
        f_list = [item for sublist in datas_list for item in sublist]
        return f_list

    def update_table(self, col_name, datas_list):
        col_idx = self.get_col_idx(col_name)
        content = []
        for i in range(self.rows_count):
            if self.content[i][col_idx] in datas_list:
                content.append(self.content[i])
        self.content = content
        self.rows_count = len(self.content)
