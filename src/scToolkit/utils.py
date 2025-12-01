"""General utility functions
"""

from . import (
    # Standard Library Imports
    copy, sub, split, Counter, re, os,
    Literal, NamedTuple, is_categorical_dtype, Sequence,
    # Sparse
    inplace_column_scale, inplace_row_scale, csr_matrix, csc_matrix, isspmatrix_csr,
    isspmatrix_csc,
    # Pandas and NumPy for data manipulation and analysis
    np, pd, nb,
    distinctipy,
    # ##########################
    # scToolkit specifics
    # Logging Functions
    get_logger)


# #########################################################
# from local files
logger = get_logger(name="utils")


# ###########################################################################################################
# Stat keeper object for nice distribution param handling
class StatKeeper:
    """Object for keeping and updating statistical data.

    The StatKeeper class tracks the minimum, maximum, mean, median, and standard
    deviation of data provided to it. It provides functionality to update these
    statistics with new data and retrieve the complete history of any specific
    statistic.

    Attributes:
        mins (list): Stores the minimum values from each update.
        maxs (list): Stores the maximum values from each update.
        means (list): Stores the mean values from each update.
        medians (list): Stores the median values from each update.
        stds (list): Stores the standard deviation values from each update.

    Returns:
        None
    """
    def __init__(self) -> None:
        """
        Initializes an empty StatKeeper object.

        The initialization sets up empty lists to track min, max, mean, median,
        and standard deviation statistics over time.
        """
        # #########################################################
        # Initialize lists to store statistical data
        self.mins = []
        self.maxs = []
        self.means = []
        self.medians = []
        self.stds = []

    def update(self, data: np.ndarray) -> None:
        """
        Update the statistical data with a new dataset.

        This method accepts a one-dimensional dataset, either as a list or a numpy array,
        and updates the stored statistics (min, max, mean, median, std) accordingly.

        Args:
            data (list or numpy.ndarray): One-dimensional data as a list or numpy array.

        Raises:
            ValueError: If data is not one-dimensional or is empty.

        TODO:
            Add dimensionality check to ensure ``data`` is strictly one-dimensional.
        """
        # #########################################################
        # Validate input data
        if not data:
            raise ValueError('Data cannot be empty.')
        # NOTE: Assumes data is a one-dimensional list or numpy array.
        # #########################################################
        # Calculate and update statistical measures
        self.mins.append(data.min())
        self.maxs.append(data.max())
        self.means.append(data.mean())
        self.medians.append(np.median(data).tolist())
        self.stds.append(data.std())

    def get(self, name: str) -> list[float]:
        """
        Retrieve the statistical data for a specified metric.

        This method returns the complete history of the specified statistic
        (min, max, mean, median, or std).

        Args:
            name (str): The name of the statistic to retrieve. Must be one of
            'min', 'max', 'mean', 'median', or 'std'.

        Returns:
            list: A list containing the history of the specified statistic.

        Raises:
            ValueError: If an unsupported statistic name is provided.
        """
        # #########################################################
        # Retrieve requested statistical data based on the name
        if name == 'median':
            return self.medians
        elif name == 'min':
            return self.mins
        elif name == 'max':
            return self.maxs
        elif name == 'mean':
            return self.means
        elif name == 'std':
            return self.stds
        else:
            # #########################################################
            # Handle unsupported statistic names
            raise ValueError('Invalid statistic name. Choose from median, min, max, mean, std.')


class FFTGrid(NamedTuple):
    """Class to store FFTGrids and associated coordinate metadata.

    This object holds:
        - Window limits (min/max)
        - Grid-to-coordinate and coordinate-to-grid scalers
        - 1D centre arrays
        - 2D meshgrid arrays
        - 1D bin edge arrays

    NOTE:
        This object enables:
            - Evaluation of functions on a fixed, uniform 2D grid.
            - Coordinate ↔ index conversions.
            - Visualization or interpolation in spatial or latent (e.g., UMAP)
              space.

    Attributes:
        x_min, x_max (float):
            Minimum and maximum x-coordinates of the window (with padding).
        y_min, y_max (float):
            Minimum and maximum y-coordinates of the window (with padding).
        x_scaler_gi_to_co, y_scaler_gi_to_co (float):
            Grid index to coordinate scalers. Convert grid index to coordinate:
            ``x_coord = x_index * x_scaler_gi_to_co + x_min``.
        x_scaler_co_to_gi, y_scaler_co_to_gi (float):
            Coordinate to grid index scalers. Convert coordinate to grid index:
            ``x_index = (x_coord - x_min) * x_scaler_co_to_gi``.
        xs, ys (np.ndarray of shape (grid_size)):
            1D coordinate arrays of bin centres for each axis.
        xx, yy (np.ndarray of shape (grid_size, grid_size)):
            2D meshgrids representing physical coordinates at grid centres.
            Suitable for visualization (e.g., ``plt.contourf(xx, yy, density)``).
        xe, ye (np.ndarray of shape (grid_size + 1)):
            1D arrays of bin edges for histograms or boundary-aware operations.

    Returns:
        None

    Tags:
        spatial, utils
    """
    # ----- geometry -----
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # Convert grid indices to real coordinates	index * x_scaler + x_min
    # Convert real coordinates to grid indices	(coord - x_min) / (x_max - x_min) * (grid_size - 1)

    x_scaler_gi_to_co: float          # (x_max - x_min) / grid_size
    y_scaler_gi_to_co: float          # (y_max - y_min) / grid_size
    x_scaler_co_to_gi: float          # (grid_size - 1) / (x_max - x_min)
    y_scaler_co_to_gi: float          # (grid_size - 1) / (x_max - x_min)
    # ----- sampling -----
    xs: np.ndarray           # 1-D x centres   (grid_size)
    ys: np.ndarray           # 1-D y centres   (grid_size)
    xx: np.ndarray           # 2-D meshgrid X  (grid_size, grid_size)
    yy: np.ndarray           # 2-D meshgrid Y  (grid_size, grid_size)
    xe: np.ndarray           # 1-D x edges     (grid_size+1)
    ye: np.ndarray           # 1-D y edges     (grid_size+1)


# class FFTGrid(NamedTuple):
#     """2-D grid & metadata with UNIT↔TISSUE (±PIXEL) transforms.

#     Spaces
#     ------
#     - UNIT  : normalized [0,1]×[0,1] space where densities are evaluated.
#     - TISSUE: physical coordinates (µm, mm, …).
#     - PIXEL : optional image coordinate system.

#     Notes
#     -----
#     - Supports **non-uniform** grids via ``xs, ys, xe, ye``.
#     - Conversions are handled via homogeneous transforms (3×3).
#     - Density arrays can be stored directly in this object.

#     Attributes
#     ----------
#     x_min, x_max, y_min, y_max : float
#         Bounds of the TISSUE window (with padding).
#     xs, ys : np.ndarray
#         1-D arrays of bin centers (strictly increasing).
#     xe, ye : np.ndarray
#         1-D arrays of bin edges (len = n_bins+1).
#     xx, yy : np.ndarray
#         2-D meshgrids of bin centers, shape (ny,nx).
#     ndim : int
#         Number of dimensions (always 2 for now).
#     density : Optional[np.ndarray]
#         Optional array of density values, shape (ny,nx).
#     A_u2t, A_t2u : np.ndarray
#         3×3 homogeneous transforms for UNIT↔TISSUE.
#     A_u2p, A_p2u : Optional[np.ndarray]
#         3×3 homogeneous transforms for UNIT↔PIXEL (None if unused).
#     """

#     # ---- geometry (TISSUE space) ----
#     x_min: float
#     x_max: float
#     y_min: float
#     y_max: float

#     # ---- sampling arrays (TISSUE) ----
#     xs: np.ndarray
#     ys: np.ndarray
#     xx: np.ndarray
#     yy: np.ndarray
#     xe: np.ndarray
#     ye: np.ndarray

#     # ---- dimension & optional density ----
#     ndim: int
#     density: Optional[np.ndarray] = None

#     # ---- transform matrices ----
#     A_u2t: np.ndarray
#     A_t2u: np.ndarray
#     A_u2p: Optional[np.ndarray] = None
#     A_p2u: Optional[np.ndarray] = None


# ###########################################################################################################
# Some libraries (e.g., decoupler) may require setting rarely used parallelization
# backendsthis function ensures all common thread settings are configured.
def set_threads(
            num_threads: int,
            set_blas_threads: bool = True,
            set_numexpr_threads: bool = True,
            set_openmp_threads: bool = True,
            set_numba_threads: bool = True,
        ) -> None:
    """
    Configure thread usage ffor numerical backends to control CPU parallelism.

    This sets environment variables for various libraries (BLAS, NumExpr,
    OpenMP, and Numba) to limit or control the number of threads used during
    computation.

    Args:
        num_threads (int): Number of threads to set for supported libraries.
        set_blas_threads (bool, optional): If True, sets thread limits for BLAS
            libraries (OpenBLAS, MKL, and VECLIB). Defaults to True.
        set_numexpr_threads (bool, optional): If True, sets thread limit for
            NumExpr. Defaults to True.
        set_openmp_threads (bool, optional): If True, sets thread limit for
            OpenMP. Defaults to True.
        set_numba_threads (bool, optional): If True, sets thread limit for
            Numba. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If ``num_threads`` is not a positive integer.

    Tags:
        utils
    """
    num_threads_str = str(num_threads)
    if not num_threads_str.isdigit():
        raise ValueError("Number of threads must be an integer.")

    if set_blas_threads:
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads_str
        os.environ["MKL_NUM_THREADS"] = num_threads_str
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads_str

    if set_numexpr_threads:
        os.environ["NUMEXPR_NUM_THREADS"] = num_threads_str

    if set_openmp_threads:
        os.environ["OMP_NUM_THREADS"] = num_threads_str

    if set_numba_threads:
        nb.set_num_threads(num_threads)


# ###########################################################################################################
# Nicer Printing handling
def print_stats(
            data: list[float] | np.ndarray,
            name: str,
            num_name: int = 10,
            prec: int = 4
        ) -> None:
    """
    Prints statistical information (mean, median, standard deviation, min,
    and max) for a given dataset.

    This function provides a summary of the central tendencies and dispersion of
    the data by printing the mean, median, standard deviation, minimum, and
    maximum values. The output is formatted with customizable precision and
    surrounded by a boundary of hashtags for emphasis.

    NOTE:
        Ensure that the 'data' input is either a list or a numpy.ndarray to
        avoid errors.

    Args:
        data (list[float] | np.ndarray): The dataset for which statistics are to
            be calculated.
        name (str): The name or label for the dataset, used in the print output.
        num_name (int, optional): The number of hashtags to print on either side
            of the name for the boundary. Defaults to 10.
        prec (int, optional): The number of decimal places to use when printing
            floating-point numbers. Defaults to 4.

    Returns:
        None

    Raises:
        TypeError: If 'data' is not a list or numpy.ndarray.

    TODO:
        Consider adding error handling for cases where 'data' contains
        non-numeric values.

    Tags:
        stats, utils
    """
    # #########################################################
    # Creating the boundary line with hashtags and printing the dataset name
    logger.info(f'{"#" * num_name} {name} {"#" * num_name}')
    # #########################################################
    # Calculating and formatting statistical values: mean, median, std, min, and max
    str_ = f"Mean: {np.mean(data):.{prec}f}, Median: {np.median(data):.{prec}f}, "
    str_ += f"Std: {np.std(data):.{prec}f}, Min: {np.min(data):.{prec}f}, Max: {np.max(data):.{prec}f}"

    # Printing the formatted statistics string
    logger.info(str_)
    # #########################################################
    # Creating and printing the closing boundary line
    logger.info(f"{'#' * ((num_name * 2) + 2 + len(name))}")


def print_stats_split(
            mean: float,
            median: float,
            std: float,
            min_v: float,
            max_v: float,
            name: str,
            num_name: int = 10,
            prec: int = 4
        ) -> None:
    """
    Prints specified statistics for a dataset, including mean, median,
    standard deviation, minimum, and maximum values, formatted with a
    specified precision and surrounded by a customizable boundary of
    hashtags.

    NOTE:
        This function directly prints the statistics and does not return any
        value.

    Args:
        mean (float): Mean value of the dataset to print.
        median (float): Median value of the dataset to print.
        std (float): Standard deviation of the dataset to print.
        min_v (float): Minimum value of the dataset to print.
        max_v (float): Maximum value of the dataset to print.
        name (str): Name or label to print as a header for the statistics.
        num_name (int, optional): Number of hashtags used for the boundary
            around the name and statistics. Defaults to 10.
        prec (int, optional): Number of decimal places to use for formatting the
            statistics. Defaults to 4.

    Returns:
        None

    Tags:
        stats, utils
    """
    # #########################################################
    # Print the formatted header with the name surrounded by hashtags
    logger.info(f'{"#" * num_name} {name} {"#" * num_name}')
    # #########################################################
    # Prepare the formatted string for the statistics
    str_ = f"Mean: {mean:.{prec}f}, Median: {median:.{prec}f}, "
    str_ += f"Std: {std:.{prec}f}, Min: {min_v:.{prec}f}, Max: {max_v:.{prec}f}"
    # #########################################################
    # Print the statistics and the closing boundary of hashtags
    logger.info(str_)
    logger.info(f"{'#' * ((num_name * 2) + 2 + len(name))}")


def get_print_highlighter(
            n: int | list[int] = 80,
            k: int = -1
        ) -> str | list[str]:
    """
    Generates a list of strings or a single string consisting of a
    repeated '#' character. The function can either return a list of strings
    with varying lengths based on the input parameters or a single string if
    specific conditions are met.

    NOTE:
        This function handles two types of inputs for 'n'. When 'n' is an
        integer and 'k' is -1, the function returns a single string. If 'k' is a
        positive integer, it returns a list of strings. When 'n' is a list, 'k'
        is ignored, and the function returns a list of strings based on the
        elements in 'n'.

    Args:
        n (int | list[int], optional): If an integer is provided, the function
            will generate a string of length 'n' consisting of '#'. If a list is
            provided, the function returns a list of strings, each string
            corresponding to the length specified by each element in the list.
            Defaults to 80.
        k (int, optional): If 'k' is set to -1 (default), the function will
            return a single string of length 'n'. If 'k' is a positive integer,
            the function will return a list of strings, where the number of
            strings is 'k' and the lengths are calculated by dividing 'n' by
            'k'. Defaults to -1.

    Returns:
        str | list[str]:
            str or list: A string if 'k' is -1 or a list of strings if 'k' is a
            positive integer or if 'n' is a list.

    Raises:
        ValueError: If 'k' is a positive integer and 'n' cannot be evenly
            divided by 'k'.

    Called By:
        run_downstream

    TODO:
        Consider handling the scenario where 'k' is a positive integer, but
        'n' is less than 'k', which could lead to unintended outputs.

    Tags:
        utils
    """
    # #########################################################
    # Check if 'n' is a list, return a list of strings with lengths
    # corresponding to the values in 'n'
    if isinstance(n, list):
        return ["#" * x for x in n]
    # #########################################################
    # Handle case where 'n' is an integer and 'k' is specified
    elif isinstance(n, int):
        # ##########################################
        # If 'k' is -1, return a single string of length 'n'
        if k == -1:
            return "#" * n
        # ##########################################
        # Otherwise, return a list of strings based on 'n' and 'k'
        else:
            div = n // k  # Calculate division result
            return ["#" * (n - (x * div)) for x in range(k)]


# ###########################################################################################################
# Dataframe Handlings
def df_split_col(
            df: pd.DataFrame,
            key: str = "genesymbol",
            sep: str = ","
        ) -> pd.DataFrame:
    """
    Split a column in a DataFrame where each cell contains a string list into
    one row per element.

    This function takes a DataFrame and a specified column containing string
    lists (e.g., "A,B,C") and splits each string list into separate rows,
    expanding the DataFrame accordingly. The operation is performed on a copy of
    the DataFrame to avoid modifying the original DataFrame.

    NOTE:
        This is the inverse function of df_group_genes().

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be split.
        key (str, optional): The name of the column to split.
            Defaults to "genesymbol".
        sep (str, optional): The delimiter used to split the string lists.
            Defaults to ".

    Returns:
        pandas.DataFrame:
            A new DataFrame with the specified column split into separate rows.

    Raises:
        KeyError: If the specified key is not found in the DataFrame's columns.

    Called By:
        get_ref_db

    Tags:
        utils
    """
    # #########################################################
    # Check if the specified column is present in the DataFrame
    if key not in df.columns:
        raise KeyError(f'The dataframe has no column {key}!')
    # #########################################################
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    # #########################################################
    # Split the specified column by the separator and expand into separate rows
    df_copy[key] = df_copy[key].str.split(sep)
    # #########################################################
    # Return the exploded DataFrame where each element of the list becomes a new row
    return df_copy.explode(key)


def df_group_genes(
            df: pd.DataFrame,
            groupby_list: str | list[str] | None = None,
            key: str = "genesymbol",
            sep: str = ",",
            join_groupby_ob: str = "|",
            joined_name: str = "cell_type",
            delete_groubpy_cols: bool = False,
        ) -> pd.DataFrame:
    """Combine a decoupler marker dataframe to have one row per geneset.

    NOTE:
        This is the inverse function of pandas explode or df_split_col().

    Args:
        df (pandas.DataFrame): The dataframe to be processed.
        groupby (str or list, optional): Columns to group by. If None, all
            columns except 'key' will be used for grouping. Defaults to None.
        key (str, optional): The name of the column to combine into a list.
            Defaults to "genesymbol".
        sep (str, optional): The separator used for joining the list elements
            into a string. Defaults to ".
        join_groupby_ob: Separator to join group labels. Defaults to "|".
        joined_name: Name of the joined label column. Defaults to "cell_type".
            Ignored if `join_groupby_ob` is an empty string.
        delete_groubpy_cols (bool, optional): Whether to drop the original
            grouping columns after creating the joined label. If True, only the
            joined label and the key column are kept. Defaults to False.

    Returns:
        pandas.DataFrame:
            The dataframe with grouped rows, where the specified key
            column is combined into a single string for each group.

    Tags:
        groupby, utils
    """
    # #########################################################
    # Determine grouping columns
    cols = list(df.keys())
    if groupby_list is None:
        groupby_list = [k for k in cols if k != key]
    elif isinstance(groupby_list, str):
        groupby_list = [groupby_list]

    # IO Checks
    # Check if all groupby keys are present and categorical
    missing = [key for key in groupby_list if key not in df.keys()]
    if missing:
        raise KeyError(f"The following keys were not found in adata.obs: {missing}")

    non_categorical = [key for key in groupby_list if df[key].dtype.name != "category"]
    if non_categorical:
        raise TypeError(
            f"The following keys are not categorical in df: {non_categorical}.\n"
            f"Dtypes were: {[df[key].dtype.name for key in non_categorical]}")
    # Ensure the key exists in the dataframe
    if key not in df.columns:
        raise KeyError(f'The dataframe has no column {key}!')
    # #########################################################
    # Create a copy of the dataframe with the list of key
    df = df.groupby(groupby_list, observed=True)[key].agg(
        lambda x: sep.join(map(str, x.dropna()))).reset_index()
    # Create one combined column for the geneset
    if join_groupby_ob:
        # if the joined_name is already present renam it to old
        if joined_name in df.columns:
            df = df.rename(columns={joined_name: f"{joined_name}_old"})
        if joined_name in groupby_list:
            groupby_list = [f"{joined_name}_old"
                            if col == joined_name else col
                            for col in groupby_list]
        # Merge the groupby into one
        df.insert(0, joined_name, df[groupby_list].astype(str).agg(
            join_groupby_ob.join, axis=1))
        # Delete the old groupby
        if delete_groubpy_cols:
            df = df[[joined_name, key]].copy()
    return df


def print_df_in_chunks(
            df: pd.DataFrame,
            chunk_size: int = 10,
            unique_values: bool = False,
            num_unique_values: int = 10
        ) -> None:
    """
    Displays a DataFrame in chunks of a specified size, optionally showing
    unique values for each column.

    This function is designed to split a DataFrame into smaller, more manageable
    chunks for easier viewing, especially when dealing with wide DataFrames. If
    ``unique_values`` is set to True, the function will display a specified number
    of unique values per column instead of the full row data.

    NOTE:
        Ensure that the DataFrame is not too large to be handled by the memory
        of your environment. It cannot handle multi-indexed dataframes

    Args:
        df (pandas.DataFrame): The DataFrame to display. This DataFrame may
            contain various data types, including strings, numbers, and
            potentially lists or dictionaries.
        chunk_size (int, optional): The number of columns to display per chunk.
            Defaults to 10.
        unique_values (bool, optional): If True, display the first
            ``num_unique_values`` unique values per column instead of the rows.
            Defaults to False.
        num_unique_values (int, optional): The number of unique values to
            display per column when ``unique_values`` is True. Defaults to 10.

    Returns:
        None

    Tags:
        utils, visualization
    """
    # #########################################################
    # Determine if the function is running inside a Jupyter Notebook
    # to handle output display accordingly.
    is_notebook = 'get_ipython' in globals()
    if is_notebook:
        # Import display function for notebook visualization
        from IPython.display import display
    # #########################################################
    # Convert any columns containing dict or list values into strings for easier display.
    df = df.applymap(lambda x: str(x) if isinstance(x, (dict, list)) else x)
    # #########################################################
    # Calculate the number of chunks needed based on chunk_size
    num_chunks = len(df.columns) // chunk_size + int(len(df.columns) % chunk_size > 0)
    # #########################################################
    # Iterate through each chunk and display it
    for i in range(num_chunks):
        # ##########################################
        # Define the start and end columns for the current chunk
        start_col = i * chunk_size
        end_col = (i + 1) * chunk_size

        # Display the current chunk's range
        logger.info(f'Displaying columns {start_col + 1} to {min(end_col, len(df.columns))}:')
        # ##########################################
        # Display either the unique values or the full rows based on the unique_values flag
        if unique_values:
            # ##########################################
            # Extract unique values for each column in the chunk
            unique_df = df.iloc[:, start_col:end_col].apply(
                lambda col: pd.Series(col.dropna().unique()[:num_unique_values]), axis=0)
            # Display unique values in a notebook or console environment
            if is_notebook:
                display(unique_df)
            else:
                logger.info(unique_df)
        else:
            # Display full rows in a notebook or console environment
            if is_notebook:
                display(df.iloc[:, start_col:end_col])
            else:
                logger.info(df.iloc[:, start_col:end_col])
        # ##########################################
        # Add a separator between chunks for clarity
        logger.info("\n" + "-" * 50 + "\n")


def clip_dataframe(
            df: pd.DataFrame,
            clip_perc: list[float],
            inplace: bool = False,
            ignore_zeros: bool = True,
            do_remove: bool = False
        ) -> pd.DataFrame:
    """Clips numeric columns of a DataFrame to specified quantile values.

    This function computes the given quantiles for all numeric columns in the
    input DataFrame and then clips the values in these columns to lie within
    these quantiles.

    Args:
        df (pandas.DataFrame): The input DataFrame containing numeric and/or
            categorical columns.
        clip_intervals : list[float, float] | None = None
            Lower and upper percentiles for clipping or removal. Defines the range
            applied by `ignore_zeros` and `do_remove`.
        inplace (bool, optional): If True, modifies the input DataFrame
            directly. If False, returns a copy of the DataFrame. Defaults to
            False.
        ignore_zeros : bool = True
            Excludes zeros when calculating quantile limits from `clip_intervals`.
            Shifts the lower bound above zero. Combined with `do_remove=True`, this
            removes zeros as out-of-range values.
        do_remove : bool = False
            Removes values outside limits defined by `clip_intervals`. If False,
            clips them to those limits. Works after `ignore_zeros` modifies the
            percentile range.

    Returns:
        pandas.DataFrame:
            A DataFrame with numeric columns clipped to the specified quantiles.

    Raises:
        ValueError: If ``clip_perc`` does not contain exactly two float values.

    Called By:
        plot_violin

    TODO:
        Validate that ``clip_perc`` values are between 0 and 1 and clip_perc[0] <
        clip_perc[1].

    Tags:
        utils
    """
    if not inplace:
        df = df.copy()

    if clip_perc is None:
        clip_perc = [0, 1]

    # Get a numeric subset
    numeric_df = df.select_dtypes(include=[np.number])
    # Subset based on ignore zeros or on all
    if ignore_zeros:
        quantiles = numeric_df.where(numeric_df != 0).quantile(clip_perc)
    else:
        quantiles = numeric_df.quantile(clip_perc)
    for col in df.select_dtypes(include=[np.number]):
        # OLD:
        # df[col] = df[col].clip(
        #     lower=quantiles.loc[clip_perc[0], col],
        #     upper=quantiles.loc[clip_perc[1], col]
        # )
        lower = quantiles.loc[clip_perc[0], col]
        upper = quantiles.loc[clip_perc[1], col]
        if do_remove:
            df[col] = df[col].where(df[col].between(lower, upper))

        else:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def save_dataframe(
            df: pd.DataFrame,
            paths: str | list[str],
            overwrite: bool = False,
            **kwargs
        ) -> None:
    """
    Saves a pandas DataFrame to one or more specified file paths,
    inferring the saving method from the file extension.

    This function acts as a wrapper around pandas' native saving methods
    (e.g., df.to_csv, df.to_excel, df.to_json, df.to_parquet). It automatically
    selects the correct saving function based on the file extension of each
    path.

    NOTE:
        df.to_excel requires an ExcelWriter for multiple sheets, but for a
        single sheet or simple saving, it can take a path directly.

    Args:
        df (pandas.DataFrame): The pandas DataFrame to be saved.
        paths (str | list[str]): A single file path (string) or a list of file
            paths (list of strings) where the DataFrame should be saved.
            The file extension (e.g., '.csv', '.xlsx', '.json') determines the
            saving format.
        overwrite (bool, optional): If True, existing files will be overwritten
            without warning. If False, and a file already exists at the
            specified path, the function will skip saving to that path and print
            a message. Defaults to False.
        **kwargs: Arbitrary keyword arguments to be passed directly to the
            underlying pandas saving function (e.g., 'index=False' for
            to_csv, 'sheet_name' for to_excel, 'indent' for to_json).

    Returns:
        None

    Examples:
        # Save to a single CSV file
        data = {'col1': [1, 2], 'col2': [3, 4]}
        test_df = pd.DataFrame(data)
        save_dataframe(test_df, "output.csv", index=False)

        # Save to a single Excel file, specifying a sheet name
        save_dataframe(test_df, "output.xlsx", sheet_name="Sheet1")

        # Save to multiple formats
        save_dataframe(test_df, ["output.csv", "output.json", "output.xlsx"], index=False)

        # Attempt to save without overwriting an existing file
        # This will skip saving if 'output.csv' already exists and overwrite=False
        save_dataframe(test_df, "output.csv", overwrite=False, index=False)

    Raises:
        ValueError: If an unsupported file extension is provided in any of the
            paths.
        TypeError: If 'paths' is neither a string nor a list of strings.

    Calls:
        StatKeeper.get

    Called By:
        create_the_hg_vcf_genesets, process_and_save_degs

    Tags:
        io
    """
    # Ensure paths is a list for consistent iteration
    if isinstance(paths, str):
        paths_list = [paths]
    elif isinstance(paths, list):
        # Validate that all items in the list are strings
        if not all(isinstance(p, str) for p in paths):
            raise TypeError("All elements in 'paths' list must be strings.")
        paths_list = paths
    else:
        raise TypeError("'paths' argument must be a string or a list of strings.")

    # Dictionary mapping file extensions to pandas saving methods
    #
    saving_methods = {
        # TODO: Add more if needed (unlikely tough)
        '.csv': df.to_csv,
        '.xlsx': df.to_excel,
        '.json': df.to_json,
        '.html': df.to_html,
        '.feather': df.to_feather,
        '.parquet': df.to_parquet,
        '.pkl': df.to_pickle,
        '.pickle': df.to_pickle}

    for path in paths_list:
        # Extract the file extension
        _, ext = os.path.splitext(path)
        ext = ext.lower()  # Convert to lowercase to handle variations like '.CSV'

        # Check if overwrite is False and the file already exists
        if not overwrite and os.path.exists(path):
            logger.info(f"Skipping '{path}': File already exists and 'overwrite' is set to False.")
            continue  # Move to the next path in the list

        # Get the appropriate saving function
        save_func = saving_methods.get(ext)

        if save_func:
            try:
                # Call the saving function with the path and any extra kwargs
                # print(f"Saving DataFrame to '{path}'...")
                if ext == '.xlsx':
                    # pandas to_excel might handle 'if_exists' depending on engine/mode,
                    # but our explicit check above handles overwrite
                    save_func(path, **kwargs)
                else:
                    save_func(path, **kwargs)
                # print(f"Successfully saved to '{path}'.")
            except Exception as e:
                logger.info(f"Error saving to '{path}': {e}")
        else:
            raise ValueError(f"Unsupported file extension: '{ext}' for path '{path}'. "
                             f"Supported extensions are: {', '.join(saving_methods.keys())}")


def get_categorical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split DataFrame columns into categorical and non-categorical lists.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        tuple[list[str], list[str]]:
            categorical_cols, non_categorical_cols

    Called By:
        plot_umap_cat_splitting

    Tags:
        utils
    """
    categorical = [col for col in df.columns if is_categorical_dtype(df[col])]
    non_categorical = [col for col in df.columns if col not in categorical]
    return categorical, non_categorical


def join_categorical_columns(
            df: pd.DataFrame,
            keys: list[str],
            new_col: str = "joined",
            inplace: bool = False,
            sep: str = "_",
        ) -> pd.DataFrame:
    """
    Join multiple categorical columns into a single string column using 'sep'
    as separator.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        keys (list[str]): List of categorical column names to join.
        new_col (str, optional): Name of the new joined column.
            Defaults to "joined".
        inplace (bool, optional): If True, modifies the input DataFrame in-place
            and returns None. Defaults to False.
        sep (str, optional): The separator between the categorical values.
            Defaults to "_".

    Returns:
        pandas.DataFrame:
            Modified DataFrame with the new joined column.

    Tags:
        utils
    """
    if not all(col in df.columns for col in keys):
        raise ValueError("One or more columns not found in DataFrame.")

    target_df = df if inplace else df.copy()
    target_df[new_col] = target_df[keys].astype(str).agg(sep.join, axis=1)

    if not inplace:
        return target_df


def get_duplicate_rows(
            df: pd.DataFrame,
            keys: Sequence[str] | None = None
        ) -> pd.DataFrame:
    """Return rows with duplicate values in given columns.

    Args:
        df: Input DataFrame.
        keys: Columns to check for duplicates.
            None: Check duplicates across all columns.
            List[str] or tuple[str, ...]: Check duplicates in these columns only.

    Returns:
        pandas.DataFrame:
            Rows that have duplicates according to the given keys.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if keys is not None and not all(col in df.columns for col in keys):
        raise ValueError("One or more keys not found in DataFrame columns")

    return df[df.duplicated(subset=keys, keep=False)]


def remove_empty_rows_and_columns(
            nparray: np.ndarray,
            empty: int = 0
        ) -> np.ndarray:
    """Removes the empty rows and columns of a numpy array.

    Empty means 0, or set empty to whatever, except for NaN.

    Args:
        nparray (numpy.ndarray): Input 2D array.
        empty (int, optional): Value to consider as empty. Defaults to 0.

    Returns:
        numpy.ndarray:
            Array with empty rows and columns removed.

    Tags:
        utils
    """
    nparray = nparray[~np.all(nparray == empty, axis=1)]
    nparray = nparray[:, ~np.all(nparray == empty, axis=0)]

    return nparray


def clean_byte_literal_strings(
            df: pd.DataFrame,
            inplace: bool = True
        ) -> pd.DataFrame:
    """Clean improperly stringified byte literals in a DataFrame.

    In bioinformatics pipelines there are rarely consistent encoding
    standards, so this function removes literal byte-string wrappers
    (e.g. "b'GATA3'") from object or string columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        inplace (bool): If True, modify df in place. Default is False.

    Returns:
        pd.DataFrame: Copy of df with cleaned string values.
    """
    target = df if inplace else df.copy()
    pattern = re.compile(r"^b'(.*)'$")

    for col in target.select_dtypes(include=["object", "string"]).columns:
        target[col] = target[col].apply(
            lambda x: pattern.sub(r"\1", x) if isinstance(x, str) else x
        )

    if not inplace:
        return target


# ###########################################################################################################
# Scipy Sparse Handlings
def inplace_max_scale_csr(
            X: csr_matrix | csc_matrix | np.ndarray,
            axis: int = 0
        ) -> None:
    """In-place max scaling of a matrix.

    For sparse matrices (CSR or CSC), applies in-place scaling along the
    specified axis using the maximum of each row or column. For dense NumPy
    arrays, performs element-wise division by the max along the chosen axis.

    Args:
        X (csr_matrix | csc_matrix | np.ndarray): Matrix to scale.
        axis (int, optional): Axis to scale (0 = columns, 1 = rows).
            Defaults to 0.

    Returns:
        None:
            Operation is performed in-place.

    Raises:
        ValueError: If ``axis`` is not 0 or 1.
        TypeError: If ``X`` is not a csr_matrix, csc_matrix, or np.ndarray.

    Called By:
        get_adata_subset

    Tags:
        scaling, sparse
    """
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    if isinstance(X, (csr_matrix, csc_matrix)):
        scale = 1 / X.max(axis=axis).toarray().flatten()
        if axis == 0:
            inplace_column_scale(X, scale)
        elif axis == 1:
            inplace_row_scale(X, scale)
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")
    elif isinstance(X, np.ndarray):
        if axis == 0:
            X /= X.max(axis=0, keepdims=True)
        elif axis == 1:
            X /= X.max(axis=1, keepdims=True)
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows).")
    else:
        raise TypeError("X must be csr_matrix, csc_matrix, or numpy.ndarray.")


def set_sparse_subset_to_zero(
            mat: csr_matrix,
            row_mask: np.ndarray,
            col_mask: np.ndarray,
            set_to: int | float = 0,
        ) -> None:
    """Sets a subset of elements in a sparse CSR matrix to zero **in-place**.

    NOTE:
        - also has csc and numpy array handlings
        - Unfortunately I didn't found any faster way,
          any conversion to lil/coo is more costy -.-

    Args:
        A (scipy.sparse.csr_matrix): The input sparse matrix in CSR format.
        row_mask (numpy.ndarray): Boolean mask array of shape (num_rows)
            indicating rows to modify.
        col_mask (numpy.ndarray): Boolean mask array of shape (num_cols)
            indicating columns to modify.
        set_to (int | float, optional): The value to set to. Defaults to 0.

    Returns:
        None:
            The function modifies the matrix in-place.

    .. doctest::

        >>> num_rows, num_cols = 5000, 5000
        >>> A = sp.random(num_rows, num_cols, density=0.1, format="csr")
        >>> row_mask = np.random.choice([True, False], size=num_rows, p=[0.5, 0.5])
        >>> col_mask = np.random.choice([True, False], size=num_cols, p=[0.5, 0.5])
        >>> set_sparse_subset_to_zero(A, row_mask, col_mask)

    Called By:
        get_adata_subset

    Tags:
        sparse, utils
    """
    if isspmatrix_csr(mat):
        # Convert row and column masks to index lists
        for i in np.where(row_mask)[0]:
            start = mat.indptr[i]
            end = mat.indptr[i + 1]
            row_cols = mat.indices[start:end]
            mask_cols = col_mask[row_cols]
            mat.data[start:end][mask_cols] = set_to

    elif isspmatrix_csc(mat):
        for j in np.where(col_mask)[0]:
            start, end = mat.indptr[j], mat.indptr[j + 1]
            rows = mat.indices[start:end]
            mask = row_mask[rows]
            mat.data[start:end][mask] = set_to
        mat.eliminate_zeros()

    elif isinstance(mat, np.ndarray):
        mat[np.ix_(row_mask, col_mask)] = set_to

    mat.eliminate_zeros()


# ###########################################################################################################
# Dict handling Helpers
def convert_int_with_na_to_string(
            df: pd.DataFrame,
            key: str,
            impute_val: int = -1234567
        ) -> pd.DataFrame:
    """
    Correctly casts a pandas DataFrame column with integer values to string,
    handling NaN values.

    This function takes a pandas DataFrame and a specified column that may
    contain NaN values. It first imputes these NaN values with a specific value,
    converts the column to integers, and then to strings. Finally, it reverts
    the imputed value back to NaN in the string format.

    NOTE:
        The impute_val can be 1e7 which would be the string '10000000'.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be
            converted.
        key (str): The name of the column to be converted.
        impute_val (int, optional): The value used to replace NaN values before
            conversion. Defaults to -1234567.

    Returns:
        pandas.DataFrame:
            The DataFrame with the specified column converted to
            string type, with original NaN values preserved.

    Raises:
        ValueError: If the specified column is not present in the DataFrame.

    Tags:
        utils
    """
    # #########################################################
    # Check if the specified column exists in the DataFrame
    if key not in df.columns:
        raise ValueError(f'column "{key}" not found in DataFrame.')
    # #########################################################
    # Impute NaN values with the specified impute value
    df[key] = df[key].fillna(impute_val)
    # #########################################################
    # Convert the column to integers and then to strings
    df[key] = df[key].astype(int).astype(str)
    # #########################################################
    # Replace the imputed value back to NaN in the string format
    df[key] = df[key].replace(str(impute_val), np.nan)
    # #########################################################
    return df


def replace_np_array_with_list_recursive(
            dict_list: dict | list
        ) -> dict | list:
    """
    Recursively replaces numpy.ndarrays with lists in nested dictionary structures.

    This function traverses a nested dictionary structure and converts any numpy
    ndarray objects it encounters into Python lists. This is particularly useful
    when you need to ensure that the data structure is JSON-serializable or when
    working in environments that do not support numpy arrays.

    NOTE:
        This function is designed to handle nested structures, where
        dictionaries may contain other dictionaries or lists, which in turn may
        contain numpy ndarrays.

    Args:
        dict_list (dict | list): A dictionary or list structure that may contain
            nested dictionaries, lists, and numpy ndarray objects. The function
            will replace all ndarray objects
        with lists in this structure.

    Returns:
        dict | list:
            dict or list: The input dictionary or list structure with all numpy
            ndarray objects replaced by Python lists. The original structure of
            the dictionary or list is preserved.

    Calls:
        replace_np_array_with_list_recursive

    Called By:
        replace_np_array_with_list_recursive, save_h5ad_old

    Tags:
        utils
    """
    # #########################################################
    # Check if the input is a dictionary to start the recursive process
    if isinstance(dict_list, dict):
        # ##########################################
        # Iterating over dictionary items and recursively replacing ndarrays
        for k, v in dict_list.items():
            dict_list[k] = replace_np_array_with_list_recursive(v)
        # Returning the modified dictionary
        return dict_list
    # #########################################################
    # Check if the input is a numpy ndarray and convert it to a list
    elif isinstance(dict_list, np.ndarray):
        return dict_list.tolist()
    # #########################################################
    # If the input is neither a dictionary nor an ndarray, return it unchanged
    else:
        return dict_list


def replace_special_chars(dictionary: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Sanitize dictionary keys by replacing Greek, special, and non-ASCII
    characters.

    Args:
        dictionary (dict[str, list[str]]): Dictionary with possibly non-ASCII
        or symbolic keys.

    Returns:
        dict[str, list[str]]:
            Dictionary with normalized ASCII-safe keys.

    Called By:
        get_ref_gensests, get_valide_ref_dicts

    Tags:
        utils
    """
    greek_to_english_and_special_chars = {
        # greek alphabet
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
        'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
        'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega', 'Α': 'Alpha',
        'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta', 'Ε': 'Epsilon', 'Ζ': 'Zeta',
        'Η': 'Eta', 'Θ': 'Theta', 'Ι': 'Iota', 'Κ': 'Kappa', 'Λ': 'Lambda',
        'Μ': 'Mu', 'Ν': 'Nu', 'Ξ': 'Xi', 'Ο': 'Omicron', 'Π': 'Pi', 'Ρ': 'Rho',
        'Σ': 'Sigma', 'Τ': 'Tau', 'Υ': 'Upsilon', 'Φ': 'Phi', 'Χ': 'Chi',
        'Ψ': 'Psi', 'Ω': 'Omega',
        # Pre defined non word chars
        "+": "_pos_", "/": "_or_",
        # Escape chars
        r"\\n": "_newline_",
        r"\\t": "_tab_",
        r"\\r": "_carriagereturn_",
        # "\\b": "_backspace_",
        r"\\f": "_formfeed_",
        r"\\v": "_verticaltab_",
        r"\\\\": "_backslash_",
        r"\\'": "_singlequote_",
        r'\\"': "_doublequote_",
        r"\\a": "_bell_",
        r"\\0": "_nullcharacter_",
        # Non ascii to ascii
        "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "Å": "A",
        "Æ": "AE", "Ç": "C", "È": "E", "É": "E", "Ê": "E", "Ë": "E",
        "Ì": "I", "Í": "I", "Î": "I", "Ï": "I", "Ð": "D", "Ñ": "N",
        "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O", "Ø": "O",
        "Ù": "U", "Ú": "U", "Û": "U", "Ü": "U", "Ý": "Y", "Þ": "TH",
        "ß": "sz", "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
        "æ": "ae", "ç": "c", "è": "e", "é": "e", "ê": "e", "ë": "e",
        "ì": "i", "í": "i", "î": "i", "ï": "i", "ð": "d", "ñ": "n",
        "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o",
        "ù": "u", "ú": "u", "û": "u", "ü": "u", "ý": "y", "þ": "th",
        "ÿ": "y"}
    new_dict = {}
    for key, value in dictionary.items():
        # Replace special characters and non-ASCII characters with underscores
        # replace the greek letters and other crap
        new_key = key
        for g, w in greek_to_english_and_special_chars.items():
            # print(g)
            new_key = sub(re.escape(g), w, new_key)
        # Relpace everythin non word or minus with _
        new_key = sub(r'[^\w-]', '_', new_key)
        # Replace all the double underscores
        new_key = sub("_", '_', new_key)
        new_dict[new_key] = value
    return new_dict


def split_overlapping_groups(
            group_dict: dict[str, Sequence[str]],
            overlap_key: str = "overlapping",
            only_overlapping: bool = False,
            sep: str = "&"
        ) -> dict[str, list[str]]:
    """Split overlapping values between groups into dedicated groups.

    Args:
        group_dict: Mapping of group names to sequences of string items.
        overlap_key: Name of the key in the output dict that will store
            all overlapping items (ignored if only_overlapping=True).
        only_overlapping: If True, create a separate key for each
            unique set of groups sharing an overlap instead of a single
            overlap_key bucket.
        sep: String used to join group names for overlap keys.

    Returns:
        dict[str, list[str]]:
            A new dict with:

                - Either:
                  - ``overlap_key`` containing all overlapping items.
                  - One new key for each unique group set with overlaps.
                - All original groups without the overlapping items.

    Raises:
        TypeError: If input is not a mapping of str -> sequence[str].
    """
    if not isinstance(group_dict, dict):
        raise TypeError("group_dict must be a mapping (dict-like).")
    if not all(
        isinstance(k, str)
        and isinstance(v, Sequence)
        and not isinstance(v, (str, bytes))
        and all(isinstance(x, str) for x in v)
        for k, v in group_dict.items()
    ):
        raise TypeError("Keys must be str, values must be sequences of str.")

    # Map each item to the groups it belongs to
    item_groups: dict[str, list[str]] = {}
    for group, items in group_dict.items():
        for item in items:
            item_groups.setdefault(item, []).append(group)

    # Find overlapping items and their group sets
    overlap_map: dict[frozenset[str], list[str]] = {}
    for item, groups in item_groups.items():
        if len(groups) > 1:
            overlap_map.setdefault(frozenset(groups), []).append(item)

    # Start with original groups minus overlaps
    new_dict: dict[str, list[str]] = {
        group: [i for i in items if len(item_groups[i]) == 1]
        for group, items in group_dict.items()
    }

    if only_overlapping:
        for group_set, items in overlap_map.items():
            name = sep.join(sorted(group_set))
            new_dict[name] = sorted(items)
    else:
        # All overlapping items in a single bucket
        overlapping_items = sorted({i for items in overlap_map.values() for i in items})
        new_dict[overlap_key] = overlapping_items

    return new_dict


# ###########################################################################################################
# List handling Helpers
def flatten_recursive(
            nested_list: list,
            uniqueify: bool = False,
            level: int | None = None,
        ) -> list:
    """Flattens a nested list into a single-level list.

    This function recursively traverses a nested list and returns a new list
    that contains all elements from the nested structure, but in a single, flat
    list. The function handles any level of nesting.

    NOTE:
        This function uses recursion and may hit a recursion limit with
        extremely deep nesting. Consider an iterative approach for deeply nested
        structures.

    Args:
        nested_list (list): A list that can contain other lists or non-list
            elements.
        uniqueify: If True, remove duplicates while preserving order.
        level: How deep to flatten.
            - None = full flatten.
            - 0 = no flatten.
            - N > 0 = flatten N levels.

    Returns:
        list:
            A flat list containing all elements from the nested structure.

    .. doctest::

        >>> flatten_recursive([1, [2, [3, 4], 5], 6])
        [1, 2, 3, 4, 5, 6]

    Raises:
        TypeError: If the input is not a list.

    Calls:
        flatten_recursive

    Called By:
        flatten_recursive

    Tags:
        utils
    """
    if not isinstance(nested_list, list):
        raise TypeError("Input must be a list")
    # #########################################################
    # Initializing the flat list to collect all elements from the nested structure
    flat_list = []
    # #########################################################
    # Loop through each item in the nested list
    for item in nested_list:
        # ##########################################
        # Check if the current item is a list itself and for recursion depth
        if isinstance(item, list) and (level is None or level > 0):
            # If it is a list, extend the flat_list with the flattened sublist
            flat_list.extend(
                flatten_recursive(
                    item,
                    uniqueify=False,
                    level=None if level is None else level - 1))
        else:
            # Otherwise, simply append the non-list item to the flat_list
            flat_list.append(item)
    # #########################################################
    # uniqueify the flat list if required
    if uniqueify:
        seen = set()
        flat_list = [x for x in flat_list if not (x in seen or seen.add(x))]
    # #########################################################
    # Return the fully flattened list
    return flat_list


# ###########################################################################################################
# Long Name Helpers
def create_acronym(
            name: str,
            ignore_numbers: bool = False
        ) -> str:
    """
    Creates an acronym from the given category name by taking the first letter
    of each word.

    This function splits the input string into words using a regular expression
    that matches any non-alphanumeric characters (including underscores). It
    then combines the first letter of each word, capitalizes it, and forms an
    acronym.

    NOTE:
        Ensure that the acronym contains only alphabetical characters.

    Token Types (after splitting on non-alphanumerics):
        A: digits only       ("2024")
        B: letters only      ("AI")
        C: neither A nor B   ("!!!")

    Token Type Presence Cases:
        1: A, B, C  -> "2024AI!!!"
        2a: A, B    -> "2024AI"
        2b: A, C    -> "2024!!!"
        2c: B, C    -> "AI!!!"
        3a: A       -> "2024"
        3b: B       -> "AI"
        3c: C       -> "!!!"

    Args:
        name (str): The category name from which to generate an acronym.
            The name can include spaces, punctuation, or underscores.
        ignore_numbers (bool, optional): If True, numbers will be excluded from
            the acronym. Defaults to False.

    Returns:
        str:
            The acronym generated from the input category name.

    Raises:
        ValueError: If the name is an empty string or None.

    Called By:
        create_unique_acronyms, plot_proximap, plot_proximap_up_down,
        plot_violin

    TODO:
        Handle edge cases with hyphenated words if necessary.

    Tags:
        utils
    """
    # #########################################################
    # Check if the input name is valid
    if not name or not isinstance(name, str):
        # Raise an error if name is empty or None
        raise ValueError("The 'name' parameter must be a non-empty string.")
    # #########################################################
    # Split the input name into words using a regular expression
    # that matches non-alphanumeric characters
    words = split(r'[\W_]+', name)
    # #########################################################
    # Create the acronym by joining the first letter of each word
    acronym = ''.join(
            word if word.isdigit() and not ignore_numbers else word[0].upper()
            for word in words if word and (not ignore_numbers or not word.isdigit()))
    # #########################################################
    return acronym


def create_unique_acronyms(
            input_data: list[str] | np.ndarray
        ) -> list[str]:
    """
    Creates unique acronyms from a list or flat numpy array of category names,
    ensuring uniqueness by appending numbers if necessary.

    Args:
        input_data (list[str] | numpy.ndarray): A list or flat numpy array of
            category names.

    Returns:
        list[str]:
            A list of unique acronyms.

    Raises:
        ValueError: If the input is not a list or flat numpy array of strings.

    Calls:
        create_acronym

    Tags:
        utils
    """
    if not isinstance(input_data, (list, np.ndarray)):
        raise ValueError("Input must be a list or flat numpy array of strings.")

    flat_data = np.array(input_data).flatten() if isinstance(input_data, np.ndarray) else input_data
    if not all(isinstance(name, str) for name in flat_data):
        raise ValueError("All elements in the input must be strings.")

    # Track occurrences of acronyms
    acronym_counts = Counter()
    unique_acronyms = []

    for name in flat_data:
        acronym = create_acronym(name)
        if acronym_counts[acronym] > 0:
            # Append a number if acronym is already used
            acronym = f"{acronym}{acronym_counts[acronym]}"
        unique_acronyms.append(acronym)
        acronym_counts[acronym] += 1

    return unique_acronyms


def ensure_unique_acronyms(
            category_map: dict
        ) -> dict:
    """
    Ensures that all acronyms in the category_map are unique by appending
    integers if necessary.

    This function takes a dictionary where the keys are categories and the
    values are acronyms. If any of the acronyms are duplicated, it appends an
    integer to make them unique. The function modifies the dictionary in-place
    and returns the updated dictionary.

    NOTE:
        The function modifies the dictionary in place, so the input
        category_map will be updated directly.

    Args:
        category_map (dict): A dictionary where the keys are categories (any
            hashable type) and values are acronyms (str). Each acronym should be
            unique, but if duplicates exist, they
        will be updated.

    Returns:
        dict:
            The updated dictionary with unique acronyms.

    Raises:
        ValueError: If the provided category_map is not a dictionary.

    Called By:
        create_acronym_column, plot_proximap, plot_proximap_up_down, plot_violin

    TODO:
        Consider adding a feature to allow customization of the separator
        between the acronym and the integer (currently it's fixed as no
        separator).

    Tags:
        utils
    """
    # #########################################################
    # Check if the input is a dictionary
    if not isinstance(category_map, dict):
        raise ValueError("Input must be a dictionary.")
    # #########################################################
    # Initialize an empty dictionary to track seen acronyms and their counts
    seen = {}

    # Iterate through the category_map to ensure unique acronyms
    for category, acronym in category_map.items():
        # ##########################################
        # If the acronym has been seen before, append an integer to make it unique
        if acronym in seen:
            # Handle duplicate acronym by appending an integer based on how many times it's seen
            count = seen[acronym]
            new_acronym = f'{acronym}_{count}'
            category_map[category] = new_acronym
            seen[acronym] += 1
        else:
            # ##########################################
            # If the acronym is unique, record it in the seen dictionary
            seen[acronym] = 1
    # #########################################################
    # Return the updated category_map with unique acronyms
    return category_map


def create_acronym_column(
            df: pd.DataFrame,
            key: str,
            out_key: str = "acronym",
            ignore_numbers: bool = False,
            unique: bool = True,
            inplace: bool = True
        ) -> pd.Series | None:
    """Creates an acronym column from a specified column in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to transform.
        key (str): The column name from which to generate acronyms.
        out_key (str, optional): The name of the output column.
            Defaults to "acronym".
        ignore_numbers (bool, optional): Whether to ignore numbers in acronym
            creation. Defaults to False.
        unique (bool, optional): Whether to ensure unique acronyms.
            Defaults to True.
        inplace (bool, optional): Whether to modify the DataFrame in place.
            Defaults to True.

    Returns:
        pandas.Series | None:
            pd.Series or None: The generated acronym column if ``inplace`` is
                False, otherwise None.

    Calls:
        ensure_unique_acronyms

    Tags:
        utils
    """
    # get the acronym
    acronym_column = df[key].map(create_acronym)
    # Uniquify the acronyms if desired
    if unique:
        acronym_column = ensure_unique_acronyms({k: v for k, v in zip(df[key].values, acronym_column)})
        acronym_column = df[key].map(acronym_column)
    # alter the input or not
    if inplace:
        df[out_key] = acronym_column
    else:
        return acronym_column


# ###########################################################################################################
# Some glob replacements for file and directory listing
def list_files(
            path: str = ".",
            pattern: str = None,
            full_names: bool = True,
            recursive: bool = False,
            include_hidden: bool = False,
            exclude_dirs: list[str] = [".ipynb_checkpoints"],
            return_absolute: bool = False
        ) -> list[str]:
    """
    List all files in a directory, with optional pattern matching and recursion.

    Args:
        path (str, optional): Directory path to search. Defaults to ".".
        pattern (str, optional): Regular expression pattern to filter file
            names. Defaults to None.
        full_names (bool, optional): If True, returns full file paths; if False,
            returns file names only. Defaults to True.
        recursive (bool, optional): If True, searches directories recursively.
            Defaults to False.
        include_hidden (bool, optional): If True, includes hidden files
            (starting with '.'). Defaults to False.
        exclude_dirs (list[str], optional): Directory names to skip (no
            wildcards, just literal names). Defaults to
            [".ipynb_checkpoints"].
        return_absolute (bool, optional): If True, returns absolute paths.
            Defaults to False.

    Returns:
        List[str]:
                List of file paths or names matching the criteria.

    Tags:
        io
    """
    files_list = []

    if recursive:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for name in files:
                if not include_hidden and name.startswith('.'):
                    continue
                full_path = os.path.join(root, name)
                if pattern is None or re.search(pattern, name):
                    if return_absolute:
                        full_path = os.path.abspath(full_path)
                    files_list.append(full_path if full_names else name)
    else:
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path):
                if not include_hidden and name.startswith('.'):
                    continue
                if pattern is None or re.search(pattern, name):
                    if return_absolute:
                        full_path = os.path.abspath(full_path)
                    files_list.append(full_path if full_names else name)

    return files_list


def list_dirs(
            path: str = ".",
            pattern: str = None,
            full_names: bool = True,
            recursive: bool = False,
            include_hidden: bool = False,
            exclude_dirs: list[str] = [".ipynb_checkpoints"],
            return_absolute: bool = False
        ) -> list[str]:
    """
    List all directories in a given path, with optional pattern matching and
    recursion.

    Args:
        path (str, optional): Directory path to search. Defaults to ".".
        pattern (str, optional): Regular expression pattern to filter directory
            names. Defaults to None.
        full_names (bool, optional): If True, returns full directory paths; if
            False, returns directory names only. Defaults to True.
        recursive (bool, optional): If True, searches directories recursively.
            Defaults to False.
        include_hidden (bool, optional): If True, includes hidden directories
            (starting with '.'). Defaults to False.
        exclude_dirs (list[str], optional): Directory names to skip (no
            wildcards, just literal names). Defaults to
            [".ipynb_checkpoints"].
        return_absolute (bool, optional): If True, returns absolute paths.
            Defaults to False.

    Returns:
        List[str]:
            List of directory paths or names matching the criteria.

    Tags:
        io
    """
    dirs_list = []

    if recursive:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for name in dirs:
                if not include_hidden and name.startswith('.'):
                    continue
                full_path = os.path.join(root, name)
                if pattern is None or re.search(pattern, name):
                    if return_absolute:
                        full_path = os.path.abspath(full_path)
                    dirs_list.append(full_path if full_names else name)
    else:
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                if not include_hidden and name.startswith('.'):
                    continue
                if pattern is None or re.search(pattern, name):
                    if return_absolute:
                        full_path = os.path.abspath(full_path)
                    dirs_list.append(full_path if full_names else name)

    return dirs_list


def get_subdirectories(
            path: str
        ) -> list[str]:
    """Gets a list of all subdirectories (recursively) from a given path.

    Args:
        path (str): The path to the directory to start searching from.

    Returns:
        list[str]:
            A list of absolute paths to all subdirectories.

    Tags:
        io
    """
    all_subdirectories = []
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            full_path = os.path.join(root, dir)
            all_subdirectories.append(full_path)

    return all_subdirectories


# ###########################################################################################################
# Others
def min_max_scale_axis(
            data: np.ndarray | pd.DataFrame = None,
            axis: int = 0,
            feature_range: tuple[float, float] = (0, 1),
            inplace: bool = False,
            data_list: list[np.ndarray | pd.DataFrame] | None = None
        ) -> np.ndarray | pd.DataFrame | list[np.ndarray | pd.DataFrame] | None:
    """Applies Min-Max scaling while preserving the original data type.

    Supports scaling a single dataset (``data``) or multiple datasets
    (``data_list``) consistently by computing a global min/max across all
    datasets.

    Args:
        data (np.ndarray | pd.DataFrame, optional): Single dataset to
            scale. Defaults to None.
        axis (int, optional): Axis along which to scale.
            - ``0``: Scale along columns (independently per column).
            - ``1``: Scale along rows (independently per row). Defaults to 0.
        feature_range (tuple[float): Target range for scaling. Defaults to (0,
            1).
        inplace (bool): If True, modifies the input data directly instead of
            returning a new object.
        data_list (list[np.ndarray | pd.DataFrame] | None):
            List of datasets to scale together using a shared global min/max.

    Returns:
        numpy.ndarray | pandas.DataFrame | list[numpy.ndarray | pandas.DataFrame] | None:
            - Returns a single scaled dataset if ``data`` is provided.
            - Returns a list of scaled datasets if ``data_list`` is provided.
            - Returns ``None`` if ``inplace=True`` (modifies the input data directly).

    .. doctest::

        >>> df1 = pd.DataFrame(np.random.rand(5, 3) * 100)
        >>> df2 = pd.DataFrame(np.random.rand(5, 3) * 100)
        >>> min_max_scale(data_list=[df1, df2], axis=0, inplace=True)  # Scales both in place

    Raises:
        ValueError: If ``feature_range`` is invalid (min >= max) or if neither
            ``data`` nor ``data_list`` is provided.

    Called By:
        plot_dotplot, plot_split_dotplot, plot_split_dotplot_mpl

    Tags:
        scaling
    """
    if feature_range[0] >= feature_range[1]:
        raise ValueError("feature_range must be (min, max) with min < max.")

    # Normalize input to a list for unified processing
    if data is not None:
        data_list = [data]
    if not data_list:
        raise ValueError("Either 'data' or 'data_list' must be provided.")
    if not inplace:
        data_list = copy(data_list)

    # Compute global min/max over all datasets
    global_min = data_list[0].min(axis=axis)
    global_max = data_list[0].max(axis=axis)
    for d in data_list[1:]:
        global_min = np.minimum(global_min, d.min(axis=axis))
        global_max = np.maximum(global_max, d.max(axis=axis))

    # Prevent division by zero
    scale = global_max - global_min
    scale[scale == 0] = 1

    if axis == 1:
        if isinstance(global_min, pd.Series):
            global_min = global_min.to_numpy()  # Convert to DataFrame for row-wise broadcasting
            scale = scale.to_numpy()

        global_min = global_min[:, np.newaxis]  # Expand for NumPy arrays
        scale = scale[:, np.newaxis]

    # Apply scaling in-place if required
    for i, d in enumerate(data_list):
        # Ensure correct shape for broadcasting
        scaled = (d - global_min) / scale
        scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

        if inplace:
            if isinstance(d, pd.DataFrame):
                d.iloc[:, :] = scaled  # Modify DataFrame in place
            else:
                d[:] = scaled  # Modify NumPy array in place
        else:
            data_list[i] = scaled

    return None if inplace else (data_list[0] if data is not None else data_list)


def get_colors_wrapped(n: int, **kwargs) -> list[tuple[float, float, float]]:
    """Returns a list of RGB colors using distinctipy.

    If n ≤ 200, returns n distinct colors.
    If n > 200, returns the first 200 distinct colors repeated cyclically.

    Args:
        n (int): Number of colors to return.
        **kwargs: parsed to distinctipy.get_colors()

    Returns:
        list[tuple[float, float, float]]:
            A list of RGB color tuples, each with values in [0.0, 1.0].

    Called By:
        plot_heatmap, set_distinct_colors

    Tags:
        visualization
    """
    if n <= 200:
        return distinctipy.get_colors(n, **kwargs)
    base_colors = distinctipy.get_colors(200, **kwargs)
    return [base_colors[i % 200] for i in range(n)]


# ###########################################################################################################
# integretry checks for functions
def validate_groupby_column(
            data: pd.DataFrame,
            groupby: str,
            check_categorical: bool = True,
            groups: list[str] | None = None,
            print_name: str = "adata.obs",
        ) -> None:
    """
    Validate that ``groupby`` exists in ``data``, is categorical if required
    and contains ``groups``.

    Args:
        data (pandas.DataFrame): DataFrame (typically ``adata.obs``) to validate.
        groupby (str): Column name to check in ``data``.
        check_categorical (bool, optional): If True, enforce categorical dtype.
            Defaults to True.
        groups (list[str] | None, optional): If given, all values must be in the
            column's categories or unique values. Defaults to None.
        print_name (str, optional): Name used in error messages.
            Defaults to "adata.obs".

    Returns:
        None

    Raises:
       KeyError: If column not in ``data``.
        TypeError: If column is not categorical (when ``check_categorical``).
        ValueError: If any ``groups`` are missing from column categories.

    Called By:
        calc_DEGs, check_group_overlap, compare_gene_expression_deltas,
        create_group_stats_df, get_data, create_random_spatial_adata,
        get_element_and_counts_with_positional_sum, get_group_hierarchy,
        get_group_key_percentages, merge_for_deg, name_groups, plot_dotplot,
        plot_embedding_density, plot_heatmap,
        plot_per_group_DEG_dotplot_n_gene_dendrogram, plot_per_group_DEG_umaps,
        plot_ripley_corrected_group_distance_matrix, plot_split_dotplot,
        plot_split_dotplot_mpl, plot_umap_sbs, plot_violin,
        score_ripley_corrected_group_distances, show_category_over_umap

    Tags:
        groupby, utils
    """
    if groupby not in data:
        raise KeyError(
            f"The specified key '{groupby}' was not found in {print_name}.keys().")
    if check_categorical and data[groupby].dtype.name != "category":
        raise TypeError(
            f"Expected '{groupby}' in {print_name} to be categorical, "
            f"but got dtype '{data[groupby].dtype.name}'.")

    if groups is not None:
        col = data[groupby]
        cats = col.cat.categories if hasattr(col, "cat") else col.unique()
        missing = [g for g in groups if g not in cats]
        if missing:
            raise ValueError(
                f"The following groups are not present in '{groupby}' of {print_name}: "
                f"{missing}")


# ###########################################################################################################
# Interpolations and FFT space functionality

# Type alias for the padding argument
PadArg = (
    float                                   # interpreted as *relative* (factor × ptp)
    | tuple[float, Literal["rel", "abs"]])   # (value, mode)


def get_fft_grid(
            data: np.ndarray,
            grid_size: int = 1024,
            pad: PadArg = (0.1, "rel"),      # default = 1 % of overall span
        ) -> FFTGrid:
    """Create a uniform 2-D grid (centres *and* edges) plus geometry meta-data.

    Args:
        data (numpy.ndarray): (n, 2) array of point coordinates.
        grid_size (int, optional): Number of centres per axis. Defaults to 1024.
        pad (PadArg, optional): Defaults to (0.1, "rel").
            Padding; either a float (relative by default) or a tuple

            If a float:
                Interpreted as a relative percentage of the data range
                (e.g., 0.02 = 2%).

            If a tuple:
                A (value, mode) pair where mode is one of:
                    - "rel": relative to span (e.g., (0.02, "rel"))
                    - "abs": absolute units (e.g., (1e-5, "abs"))

    Returns:
        FFTGrid:
            Named tuple containing window limits, scalers, 1D centres, 2D mesh,
            and bin edges.

            - x_min, x_max, y_min, y_max: float window boundaries
            - x_scaler_gi_to_co, y_scaler_gi_to_co: grid index to coordinate
              scalers
            - x_scaler_co_to_gi, y_scaler_co_to_gi: coordinate to grid index
              scalers
            - xs, ys: 1D coordinate arrays of bin centres
            - xx, yy: 2D meshgrid coordinate arrays
            - xe, ye: 1D bin edge arrays (length = grid_size + 1)

    Called By:
        Spatial_prox.set_min_max, interpolate_density_to_embedding

    Tags:
        spatial
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must be an (n, 2) array")

    # interpret padding argument
    if isinstance(pad, tuple) or isinstance(pad, list):
        value, mode = pad
    else:  # plain float to relative factor
        value, mode = pad, "rel"

    if mode == "rel":
        pad_val = value * np.ptp(data)
    elif mode == "abs":
        pad_val = value
    else:
        raise ValueError("pad mode must be 'rel' or 'abs'")

    # Window limits with tiny absolute padding
    x_min, y_min = data.min(axis=0) - pad_val
    x_max, y_max = data.max(axis=0) + pad_val
    # Per-axis scalers (index to original units)
    x_scaler_gi_to_co = (x_max - x_min) / grid_size
    y_scaler_gi_to_co = (y_max - y_min) / grid_size
    # Per-axis scalers (index to original units)
    x_scaler_co_to_gi = (grid_size - 1) / (x_max - x_min)
    y_scaler_co_to_gi = (grid_size - 1) / (y_max - y_min)

    # 1-D centre coordinates
    xs = np.linspace(x_min, x_max, grid_size, dtype=float)
    ys = np.linspace(y_min, y_max, grid_size, dtype=float)

    # Efficient 2-D centre mesh
    xx, yy = np.mgrid[
        x_min:x_max:grid_size * 1j,
        y_min:y_max:grid_size * 1j]

    # Histogram/DCT edges
    xe = np.linspace(x_min, x_max, grid_size + 1, dtype=float)
    ye = np.linspace(y_min, y_max, grid_size + 1, dtype=float)

    return FFTGrid(
        x_min, x_max, y_min, y_max,
        x_scaler_gi_to_co, y_scaler_gi_to_co,
        x_scaler_co_to_gi, y_scaler_co_to_gi,
        xs, ys, xx, yy, xe, ye)


def bilinear_interpolate_numpy(
            grid: np.ndarray,
            x: np.ndarray,
            y: np.ndarray
        ) -> np.ndarray:
    """
    Perform bilinear interpolation on a 2D grid for given x and y arrays
    (in pixel coordinates).

    Args:
        grid (numpy.ndarray): 2D array representing the grid values.
        x (numpy.ndarray): Array of x coordinates (column indices).
        y (numpy.ndarray): Array of y coordinates (row indices).

    Returns:
        numpy.ndarray:
            Interpolated values at the given (x, y) coordinates.

    Called By:
        interpolate_density_to_embedding

    Tags:
        utils
    """
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip to bounds
    x0 = np.clip(x0, 0, grid.shape[1] - 1)
    x1 = np.clip(x1, 0, grid.shape[1] - 1)
    y0 = np.clip(y0, 0, grid.shape[0] - 1)
    y1 = np.clip(y1, 0, grid.shape[0] - 1)

    # Gather values
    Ia = grid[y0, x0]
    Ib = grid[y1, x0]
    Ic = grid[y0, x1]
    Id = grid[y1, x1]

    # Calculate weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


# ###########################################################################################################
# Parallelizations or jit speedups
@nb.njit
def rankdata_numba(arr: np.ndarray) -> np.ndarray:
    """Assign ranks to data with ties averaged.

    Args:
        arr (numpy.ndarray): Input array to rank.

    Returns:
        numpy.ndarray:
            Array of ranks, with tied values assigned their average rank.

    Called By:
        aREA, get_inter_pvals, get_inter_pvals_old, parallel_rank

    Tags:
        calculation
    """
    n = len(arr)
    ranks = np.empty(n, dtype=np.float64)
    sorter = np.argsort(arr)
    inv = np.empty(n, dtype=np.int64)
    inv[sorter] = np.arange(n)

    i = 0
    while i < n:
        start = i
        while i + 1 < n and arr[sorter[i]] == arr[sorter[i + 1]]:
            i += 1
        average_rank = 0.5 * (start + i) + 1
        for j in range(start, i + 1):
            ranks[sorter[j]] = average_rank
        i += 1

    return ranks


@nb.njit(parallel=True, fastmath=True)
def parallel_rank(
            array: np.ndarray,
            axis: int = 0
        ) -> np.ndarray:
    """Parallel and numba compliant ranking of rows or columns

    Args:
        array (numpy.ndarray): Input array to be ranked.
        axis (int, optional): Axis to rank along (0 for columns, 1 for rows).
            Defaults to 0.

    Returns:
        numpy.ndarray:
            Array of ranks corresponding to the input array.

    Calls:
        rankdata_numba

    Called By:
        aREA, get_proximity_based_score_ranks

    .. code-block:: python

        >>> dim = 10000
        >>> # Generate random float array
        >>> np.random.seed(42)
        >>> random_array = np.random.rand(dim, dim)
        >>> # Rank using Numba function
        >>> %timeit numba_ranked = parallel_rank(random_array, axis=1)
        >>> # Rank using SciPy function
        >>> %time scipy_ranked = np.apply_along_axis(rankdata, axis=1, arr=random_array)
        >>> # Check if the results match
        >>> print(np.allclose(numba_ranked, scipy_ranked, atol=1e-300))
        >>> %timeit numba_ranked = parallel_rank(random_array, axis=0)
        >>> # Rank using SciPy function
        >>> %timeit scipy_ranked = np.apply_along_axis(rankdata, axis=0, arr=random_array)
        >>> # Check if the results match
        >>> print(np.allclose(numba_ranked, scipy_ranked, atol=1e-300))
        >>> # Check 1d case:
        >>> numba_ranked = parallel_rank(random_array[0,:])
        >>> scipy_ranked = rankdata(random_array[0,:])
        >>> print(np.allclose(numba_ranked, scipy_ranked, atol=1e-300))
        >>> numba_ranked = parallel_rank(random_array[:,0])
        >>> scipy_ranked = rankdata(random_array[:,0])
        >>> print(np.allclose(numba_ranked, scipy_ranked, atol=1e-300))

    Tags:
        calculation
    """
    # ###############################
    # Handle 1D case directly
    if array.ndim == 1:
        return rankdata_numba(array)
    # ###############################
    # Rank along columns
    if axis == 0:
        ranked_array = np.empty_like(array, dtype=np.float64)
        for j in nb.prange(array.shape[1]):
            ranked_array[:, j] = rankdata_numba(array[:, j])
    # ###############################
    # Rank along rows
    elif axis == 1:
        ranked_array = np.empty_like(array, dtype=np.float64)
        for i in nb.prange(array.shape[0]):
            ranked_array[i, :] = rankdata_numba(array[i, :])
    else:
        raise ValueError("Axis must be 0 or 1.")

    return ranked_array


@nb.njit
def min_max_scaler(data: np.ndarray) -> np.ndarray:
    """Apply min-max normalization to the input array.

    Args:
        data (numpy.ndarray): Input array to be scaled.

    Returns:
        numpy.ndarray:
            Scaled array with values in the range [0, 1].

    Called By:
        apply_density_correction

    Tags:
        scaling
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


@nb.njit
def determine_clip_threshold(
            data: np.ndarray,
            z_score: float = 2.807
        ) -> float:
    """
    Compute clipping threshold using log1p and manual z-score (numba-compatible).

    Args:
        data (numpy.ndarray): Raw data values.
        z_score (float, optional): z-score to clip to. Defaults to 99.5% = 2.807.

    Returns:
        float:
            Clipping threshold.

    Called By:
        calc_hash_features, fix_hash_out_of_distribution

    Tags:
        calculation
    """
    log_data = np.log1p(data)
    mean = np.mean(log_data)
    std = np.std(log_data)

    if std == 0.0:
        # No variation, no outliers
        return np.max(data) + 1

    log_z = (log_data - mean) / std

    has_outliers = False
    min_val = np.max(data) + 1

    for i in range(len(data)):
        if log_z[i] > z_score:
            if not has_outliers or data[i] < min_val:
                min_val = data[i]
                has_outliers = True

    if has_outliers:
        return min_val
    else:
        return np.max(data) + 1


@nb.njit(parallel=True, fastmath=True)
def _compute_ratio_csr(
            data: np.ndarray,
            indptr: np.ndarray,
            n_highest: int,
        ) -> np.ndarray:

    """Compute per-row ratio of top-n counts vs. remaining counts.

    This function is optimized for compressed sparse row (CSR) matrices.
    Each row represents a cell, and nonzero values are gene counts.
    For each row, the top ``n_highest`` values are summed and divided
    by the sum of all remaining values. The function is parallelized
    across rows using Numba.

    Args:
        data (np.ndarray): Flattened nonzero values of the CSR matrix.
        indptr (np.ndarray): CSR index pointer array of shape (n_cells + 1,).
            Defines row boundaries in ``data``.
        n_highest (int): Number of largest values per row to include
            in the numerator.

    Returns:
        np.ndarray: Ratios for each row (n_cells,). Each entry is:
            - ``np.nan`` if the row is empty.
            - ``np.inf`` if the row has no remainder beyond the top-n.
            - Otherwise, ``sum(top-n) / sum(rest)``.

    Notes:
        - Sorting is performed per row in descending order.
        - For large rows and small ``n_highest``, replacing ``np.sort``
          with ``np.partition`` may be more efficient.
    """
    n_cells = indptr.shape[0] - 1
    ratios = np.empty(n_cells, dtype=np.float64)

    for i in nb.prange(n_cells):
        start = indptr[i]
        end = indptr[i + 1]

        row_data = data[start:end]

        if row_data.size == 0:
            ratios[i] = np.nan
            continue

        # Sort descending
        sorted_vals = np.sort(row_data)[::-1]

        top_sum = sorted_vals[:n_highest].sum()
        rest_sum = sorted_vals[n_highest:].sum()

        ratios[i] = 100 * top_sum / (top_sum + rest_sum)

    return ratios


# ######################
# Jaccard
# --- Numba version ---
@nb.njit
def get_run_lengths(arr: np.ndarray) -> np.ndarray:
    """Compute the run lengths of consecutive identical values.

    Args:
        arr (numpy.ndarray): Input array of values.

    Returns:
        numpy.ndarray:
            Array of (value, count) pairs for each run, sorted by value.

    Called By:
        categorically_group_equal_numba

    Tags:
        calculation
    """
    if len(arr) == 0:
        return np.empty((0, 2), dtype=np.int64)
    runs = []
    last = arr[0]
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == last:
            count += 1
        else:
            runs.append((last, count))
            last = arr[i]
            count = 1
    runs.append((last, count))
    return np.array(sorted(runs), dtype=np.int64)


@nb.njit
def counts_equal(
            a: np.ndarray,
            b: np.ndarray
        ) -> bool:
    """Check if two arrays have the same count of each unique element.

    Args:
        a (numpy.ndarray): First input array of integer values.
        b (numpy.ndarray): Second input array of integer values.

    Returns:
        bool:
            True if both arrays have equal counts for all unique elements,
            otherwise False.

    Called By:
        categorically_group_equal_numba

    Tags:
        calculation
    """
    max_val = max(np.max(a), np.max(b)) + 1
    count_a = np.zeros(max_val, dtype=np.int64)
    count_b = np.zeros(max_val, dtype=np.int64)
    for i in range(len(a)):
        count_a[a[i]] += 1
        count_b[b[i]] += 1
    return np.array_equal(count_a, count_b)


@nb.njit
def categorically_group_equal_numba(
            a: np.ndarray,
            b: np.ndarray
        ) -> bool:
    """
    Check if two arrays are categorically equal by length and value
    frequency.

    Args:
        a (numpy.ndarray): First input array of integer values.
        b (numpy.ndarray): Second input array of integer values.

    Returns:
        bool:
            True if arrays have the same length and identical counts of unique
            values, otherwise False.

    Calls:
        counts_equal, get_run_lengths

    TODO:
        The implementation is incomplete and may return None. Final return
        condition needs to be implemented.

    Tags:
        utils
    """
    if len(a) != len(b):
        return False
    if not counts_equal(a, b):
        return False
    return (
        np.array_equal(get_run_lengths(a), get_run_lengths(b))
        or np.array_equal(get_run_lengths(a), get_run_lengths(b[::-1])))


"""
folder_path = "$GIT_PATH/scToolkit/src/" # Replace with your folder path
total_lines = count_code_lines(folder_path, list_files=True)

print(f"Total number of code lines: {count_code_lines(folder_path)}")

sc_plots.py: 2082 lines
common_logger.py: 87 lines
decoupler_speedup.py: 357 lines
__init__.py: 139 lines
sc_spatial.py: 4269 lines
markers.py: 59 lines
sc_utils.py: 3418 lines
count_lines.py: 44 lines
sc_replacements.py: 93 lines
sc_legacy.py: 172 lines
paths.py: 61 lines
utils.py: 1153 lines
sc_config.py: 717 lines
sc_code.py: 1570 lines
test.py: 107 lines
cellranger.py: 275 lines

Total number of code lines: 14603
"""
