import pandas as pd
import numpy as np
import re
import warnings

class BaseDataHandler():
    '''
    A base class for handling and manipulating pandas DataFrames.
    '''

    def __init__(self, path:str | None = None, df:pd.DataFrame | None = None, encoding:str='utf-8'):
        
        self.file_path = path
        self.encoding = encoding
        success, e = self.try_init_df(df)
        if not success:
            print(e)
    
    @property
    def og_df(self) -> pd.DataFrame:
        return self.__df
    
    @property
    def df(self) -> pd.DataFrame:
        return self.__curr_df
    
    @df.setter
    def df(self, df) -> None:
        self.__curr_df = df
    
    def df_log(self, base:float='e') -> pd.DataFrame:
        '''
        Apply logarithmic transformation to numeric columns of the DataFrame.
        '''
        numeric_cols = self.df.select_dtypes(include='number').columns
        df_log = self.df.copy()

        if base == 'e':
            df_log[numeric_cols] = df_log[numeric_cols].apply(lambda x: np.log(x.clip(lower=1e-9)))
        else:
            df_log[numeric_cols] = df_log[numeric_cols].apply(lambda x: np.log(x.clip(lower=1e-9)) / np.log(base))

        return df_log

    def get_lines(self, amount=5) -> pd.DataFrame:
        '''
        Get the first or last 'amount' of lines from the original DataFrame.
        '''
        return self.og_df.head(amount) if amount > 0 else self.og_df.tail(amount)

    def print_dataframe(self, full=False):
        '''
        Print the original DataFrame, either fully or just the first 5 rows.
        '''
        row = self.og_df.shape[0] if full else 5
        return self.og_df.head(row)
    
    def get_pivot(self, values=None, index=None, columns=None, aggfunc:str="mean") -> pd.DataFrame:
        '''
        Create a pivot table from the original DataFrame.
        '''
        return pd.pivot_table(self.og_df, values=values, index=index, columns=columns, aggfunc=aggfunc)
    
    def try_get_groupby(self, target_col: str | list[str], col:str) -> tuple[bool, any]:
        '''
        Group the original DataFrame by 'target_col' and select 'col'.
        '''
        try:
            tmp_df = self.og_df.groupby(by=target_col)[col].count()
        except Exception as e:
            return False, e
        return True, tmp_df

    def try_init_df(self, df) -> tuple[bool, any]:
        '''
        Initialize the DataFrame and working DataFrame from a given DataFrame or by reading from a CSV file.
        '''
        try:
            if df is not None:
                self.__df = df
                self.__curr_df = df
            else:
                self.__df = pd.read_csv(self.file_path, encoding=self.encoding)
                self.__curr_df = self.og_df.copy()
        except Exception as e:
            return False, e
        return True, None
    
    def try_reset_df(self) -> tuple[bool, any]:
        '''
        Reset the working DataFrame to the original DataFrame.
        '''
        try:
            self.__curr_df = self.og_df.copy()
        except Exception as e:
            return False, e
        return True, None
    
    def try_update_og_df(self) -> tuple[bool, any]:
        '''
        Update the original DataFrame to match the current working DataFrame.
        '''
        try:
            self.__df = self.__curr_df.copy()
        except Exception as e:
            return False, e
        return True, None
    
    def try_order_by(self, cols:str | list[str], ascending:bool | list[bool]=True) -> tuple[bool, any]:
        '''
        Order the original DataFrame by specified columns.
        '''
        try:
            self.__curr_df = self.og_df.sort_values(by=cols, ascending=ascending).reset_index()
        except Exception as e:
            return False, e
        return True, None

    def try_fill_nan(self, use_mean:bool = True) -> tuple[bool, any]:
        '''
        Fill NaN values in the original DataFrame with either 0 or the mean of each column.
        '''
        try:
            self.__curr_df = self.og_df.fillna(0 if use_mean else self.og_df.mean(numeric_only=True))
        except Exception as e:
            return False, e
        return True, None
    
    def try_add_col(self, target_col:str, criteria, axis:int=1) -> tuple[bool, any]:
        '''
        Add a new column to the working DataFrame based on a criteria function applied to each row or column.
        '''
        try:
            self.__curr_df[target_col] = self.og_df.apply(criteria, axis=axis)
        except Exception as e:
            return False, e
        return True, None
    
    def try_remove_duplicates(self) -> tuple[bool, any]:
        '''
        Remove duplicate rows from the original DataFrame.
        '''
        try:
            self.__curr_df = self.og_df.drop_duplicates()
        except Exception as e:
            return False, e
        return True, None
    
    def try_save(self) -> tuple[bool, any]:
        '''
        Save the original DataFrame to a new CSV file.
        '''
        try:
            new_file_path = self.file_path.replace('.csv', '_new.csv')
            self.og_df.to_csv(new_file_path)
        except Exception as e:
            return False, e
        return True, None
    
    def try_drop_nan(self, cols:str|list[str]) -> tuple[bool, any]:
        '''
        Drop rows with NaN values in specified columns and drop columns that are entirely NaN.
        '''
        try:
            self.__curr_df = self.og_df.dropna(subset=cols)
            self.__curr_df = self.og_df.dropna(axis=1, how='all')
        except Exception as e:
            return False, e
        return True, None
    
    def try_clamp_cols(self, cols:str|list[str], lower_bounds:float=0, upper_bounds:float=200, use_og:bool = False) -> tuple[bool, any]:
        '''
        Clamp the values in specified columns to be within given lower and upper bounds.
        '''
        try:
            self.__curr_df[cols] = self.og_df[cols].clip(lower_bounds,upper_bounds) if use_og else self.__curr_df[cols].clip(lower_bounds,upper_bounds)
        except Exception as e:
            return False, e
        return True, None
    
    def df_norm(self, method:str="minmax") -> pd.DataFrame:
        '''
        Normalize numeric columns of the DataFrame using specified method ('minmax' or 'zscore').
        '''
        numeric_cols = self.df.select_dtypes(include='number').columns
        df_norm = self.df.copy()
        if method == "minmax":
            df_norm[numeric_cols] = df_norm[numeric_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
        elif method == "zscore":
            df_norm[numeric_cols] = df_norm[numeric_cols].apply(
                lambda x: (x - x.mean()) / x.std(ddof=0)
            )
        return df_norm

    def try_clean_string_to_number_col(self, col: str | list[str]) -> tuple[bool, any]:
        '''
        Clean a column containing string representations of numbers by extracting numeric values
        and converting them to floats. Handles ranges, plus signs, decimals, and negatives.
        '''
        def helper(s):
            s = str(s).replace(",", "")
            # Match integers, decimals, and negatives
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if nums:
                nums = list(map(float, nums))
                return float(np.mean(nums))
            return None

        try:
            if isinstance(col, str):
                self.__curr_df[col] = self.df[col].apply(helper)
            elif isinstance(col, list):
                for c in col:
                    self.__curr_df[c] = self.df[c].apply(helper)
        except Exception as e:
            return False, e
        return True, None
    
    def try_clean_column_names(self) -> tuple[bool, any]:
        '''
        Clean column names by converting them to snake_case.
        '''
        def to_snake(name: str) -> str:
            # Lowercase
            name = name.lower()
            # Replace non-alphanumeric with underscore
            name = re.sub(r'[^a-z0-9]+', '_', name)
            # Remove leading/trailing underscores
            name = name.strip('_')
            return name
        try:
            self.__curr_df = self.df.rename(columns={col: to_snake(col) for col in self.df.columns})
        except Exception as e:
            return False, e
        return True, self.df
    
    def try_rename_col(self, col: str | list[str], name: str | list[str]) -> tuple[bool, any]:
        '''
        Rename specified columns in the DataFrame.
        '''
        try:
            # Ensure both are lists for mapping
            if isinstance(col, str) and isinstance(name, str):
                mapping = {col: name}
            elif isinstance(col, list) and isinstance(name, list):
                mapping = dict(zip(col, name))
            self.__curr_df = self.df.rename(columns=mapping)
            return True, self.df
        except Exception as e:
            return False, e

    import pandas as pd

    def detect_outliers_all(self, method: str = "iqr") -> pd.DataFrame:
        """
        Detect outliers for every numeric column in the DataFrame.
        Returns a DataFrame with boolean flags for each column.
        """
        outlier_flags = pd.DataFrame(index=self.df.index)

        for col in self.df.select_dtypes(include="number").columns:
            series = self.df[col]

            if method == "zscore":
                z_scores = (series - series.mean()) / series.std()
                outlier_flags[col + "_is_outlier"] = (z_scores.abs() > 3)

            else:
                if method != "iqr":
                    warnings.warn("Defaulting to IQR method for outlier detection.", UserWarning)
                Q1, Q3 = series.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outlier_flags[col + "_is_outlier"] = (series < lower) | (series > upper)

        return outlier_flags    
    
