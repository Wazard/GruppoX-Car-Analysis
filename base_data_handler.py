import pandas as pd


class BaseDataHandler():
    """Lightweight dataframe helper for CSV-backed data.

    This class centralises simple operations on a pandas DataFrame that is
    usually read from a CSV file. Methods follow a "try_*" pattern and return
    a tuple of (success: bool, result_or_exception).

    Attributes:
        file_path (str|None): path to the CSV file (if provided)
    """

    def __init__(self, path: str | None = None, df: pd.DataFrame | None = None):
        """Initialize the handler.

        Either `df` may be provided directly, or `path` must point to a CSV
        file that will be read into `self.__df` via `try_update_df`.

        Args:
            path: optional path to a CSV file.
            df: optional pre-built pandas DataFrame to use instead of reading a file.
        """

        self.file_path = path

        success, e = self.try_update_df(df)
        if not success:
            print(e)

    @property
    def df(self) -> pd.DataFrame:
        """Return the internal pandas DataFrame.

        Returns:
            pd.DataFrame: the currently-stored DataFrame.
        """

        return self.__df

    def get_lines(self, amount=5) -> pd.DataFrame:
        """Return top or bottom rows from the DataFrame.

        Positive `amount` returns that many rows from the top (head). A
        non-positive `amount` returns rows from the bottom via `tail`.

        Args:
            amount: number of rows to return (default 5). If <= 0, returns tail.

        Returns:
            pd.DataFrame: subset of rows.
        """

        return self.df.head(amount) if amount > 0 else self.df.tail(amount)

    def print_dataframe(self):
        """Print the current DataFrame to standard output.

        This is a convenience wrapper around `print(self.df)` for quick
        interactive inspection.
        """

        print(self.df)

    def get_pivot(self, values=None, index=None, columns=None, aggfunc: str = "mean") -> pd.DataFrame:
        """Create and return a pivot table from the DataFrame.

        Args:
            values: column(s) to aggregate.
            index: index (rows) for the pivot.
            columns: columns for the pivot.
            aggfunc: aggregation function to use (default: 'mean').

        Returns:
            pd.DataFrame: pivot table constructed from `self.df`.
        """

        return pd.pivot_table(self.df, values=values, index=index, columns=columns, aggfunc=aggfunc)

    def try_get_groupby(self, target_col: str | list[str], col: str) -> tuple[bool, any]:
        """Try to create a GroupBy object for the given column(s).

        This returns a tuple `(success, result)` where `result` is either the
        grouped object (on success) or the exception instance (on failure).

        Args:
            target_col: column name or list of column names to group by.
            col: column to select from the grouped object.

        Returns:
            tuple[bool, any]: (True, grouped_column) on success, (False, exception) on error.
        """

        try:
            tmp_df = self.df.groupby(by=target_col)[col]
        except Exception as e:
            return False, e
        return True, tmp_df

    def try_update_df(self, df) -> tuple[bool, any]:
        """Set the internal DataFrame from `df` or by reading `self.file_path`.

        If `df` is provided it becomes the internal DataFrame. Otherwise the
        method attempts to read a CSV from `self.file_path`.

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on failure.
        """

        try:
            if df is not None:
                self.__df = df
            else:
                self.__df = pd.read_csv(self.file_path)
        except Exception as e:
            return False, e
        return True, None

    def try_order_by(self, cols: str | list[str], ascending: bool | list[bool] = True) -> tuple[bool, any]:
        """Sort the internal DataFrame by one or more columns.

        The method replaces `self.__df` with a sorted and reindexed DataFrame.

        Args:
            cols: column name or list of column names to sort by.
            ascending: boolean or list of booleans for sort direction.

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on error.
        """

        try:
            self.__df = self.df.sort_values(by=cols, ascending=ascending).reset_index()
        except Exception as e:
            return False, e
        return True, None

    def try_fill_nan(self, use_mean: bool = True) -> tuple[bool, any]:
        """Fill NaN values in the DataFrame.

        Note: current implementation uses `0` when `use_mean` is True and uses
        the per-column numeric mean when `use_mean` is False. The docstring
        describes the implemented behaviour (no logic changes made).

        Args:
            use_mean: when True fills with 0; when False fills with the numeric column means.

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on failure.
        """

        try:
            self.__df = self.df.fillna(0 if use_mean else self.df.mean(numeric_only=True))
        except Exception as e:
            return False, e
        return True, None

    def try_add_col(self, target_col: str, criteria, axis: int = 1) -> tuple[bool, any]:
        """Add a new column to the DataFrame derived from `criteria`.

        The `criteria` callable is applied to each row/column (depending on
        `axis`) and the result is assigned to `target_col`.

        Args:
            target_col: name of the column to add or overwrite.
            criteria: callable used in `DataFrame.apply` to compute the column values.
            axis: axis value forwarded to `apply` (default 1, apply per-row).

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on error.
        """

        try:
            self.__df[target_col] = self.df.apply(criteria, axis=axis)
        except Exception as e:
            return False, e
        return True, None

    def try_remove_duplicates(self) -> tuple[bool, any]:
        """Remove duplicate rows from the DataFrame.

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on failure.
        """

        try:
            self.__df = self.df.drop_duplicates()
        except Exception as e:
            return False, e
        return True, None

    def try_save(self) -> tuple[bool, any]:
        """Save the current DataFrame to a new CSV file.

        The new file path is produced by replacing the first occurrence of
        `.csv` in `self.file_path` with `_new.csv`.

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on error.
        """

        try:
            new_file_path = self.file_path.replace('.csv', '_new.csv')
            self.df.to_csv(new_file_path)
        except Exception as e:
            return False, e
        return True, None

    def try_drop_nan(self, cols: str | list[str]) -> tuple[bool, any]:
        """Drop rows with NaNs in `cols` then drop entirely-empty columns.

        Args:
            cols: column name or list of column names to consider when dropping rows.

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on error.
        """

        try:
            self.__df = self.df.dropna(subset=cols)
            self.__df = self.df.dropna(axis=1, how='all')
        except Exception as e:
            return False, e
        return True, None

    def try_clamp_cols(self, cols: str | list[str], min_v: float = 0, max_v: float = 200) -> tuple[bool, any]:
        """Clamp numeric values in `cols` to the interval [min_v, max_v].

        Args:
            cols: column name or list of columns to clip.
            min_v: minimum allowed value (inclusive).
            max_v: maximum allowed value (inclusive).

        Returns:
            tuple[bool, any]: (True, None) on success, (False, exception) on error.
        """

        try:
            self.__df[cols] = self.df[cols].clip(min_v, max_v)
        except Exception as e:
            return False, e
        return True, None