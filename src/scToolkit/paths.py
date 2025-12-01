'''This file saves the paths and checks if the default paths are correct.'''

from . import (
    os, pd, re, __path__, ast)

# ########################################################################################
# Set the path according to this repository
__database__ = __path__[0].replace("src/scToolkit", "")
path_to_repo = __database__  # "/home/human/repos/test/minimalsphinx-main/"
# ########################################################################################
# We don't trust anybody so we check again
# ########################################################################################
if path_to_repo[-1] != "/":
    path_to_repo = path_to_repo + "/"
# ########################################################################################
# Create default structure
# ########################################################################################
# Setup the necessary paths
ALL_PATHS = {
        # This is the path to the scToolkit folder
        "PATH_TO_REPO": path_to_repo,
        # This is the path to the database folder for the QC and marker genesets
        "PATH_TO_DATABASE": path_to_repo + "databases/"}

# Here you can add the genesets for your organism or add/adjust/update files for the automatic processing
GENESET_FILENAMES = {
        # NOTE: the paths in PATH_TO_QC_GENESETS have to be relative to PATH_TO_DATABASE!!!
        # NOTE: The files should contain one geneset per row with empty strings not NA!
        # NOTE: If you add new qc genesets make sure to only add the usefull ones
        #       in the config["general"]["qc_keys_to_keep"]["obs"] in the sc_config.py!
        # HIGHLIGHTER: XXXDATABASE_UPDATEXXX
        "PATH_TO_QC_GENESETS": {
            "human": {
                "HK": "human_housekeeping_genesets.zip",
                "GTF": "GRCh38_112_gene_to_function_processed.zip",
                "HGNC": "HGNC_geneset_categories.zip"},
            "mouse": {
                "HK": "mouse_housekeeping_genesets.zip",
                "GTF": "GRCm39_112_gene_to_function_processed.zip",
                "MGI": "MGI_geneset_categories.zip"}},
        "PATH_TO_CELL_TYPE_GENESETS": {
            "human": {
                "PanglaoDB": "panglaoDB.zip",
                "CellTypist": "CellTypist_v2.zip",
                "CellMarker": "Cell_marker_All.zip",
                "SCType": "ScTypeDB_full_spaceless.zip",
            },
            "mouse": {
                "PanglaoDB": "panglaoDB.zip",
                "CellMarker": "Cell_marker_All.zip",
            }},
        "PATH_TO_GENESETS_DATABASES": {
            "human": {
                "msigdb": "msigdb/msigdb.v2024.1.Hs.json"
            },
            "mouse": {
                "msigdb": "msigdb/msigdb.v2024.1.Mm.json"
            }},
        "PATH_TO_RECEPTOR_LIGAND_GENESETS": {
            "human": {"OmniPath": "omnipath_LR_whole.csv.zip"},
            "mouse": {}
        }}

# ########################################################################################
# Check if the paths are valide
# ########################################################################################
# Check if the path is correctly set and correct it if necessary and check if they exist!
for k, v in ALL_PATHS.items():
    if v[-1] != "/":
        ALL_PATHS[k] = v + "/"
        if not os.path.exist(ALL_PATHS[k]):
            raise FileNotFoundError(
                f'{ALL_PATHS[k]} does not exist, please check if you set up '
                "the paths in the\n scToolkit correctly!!!")

# Check if the genesets exist
for k, organisms in GENESET_FILENAMES.items():
    for organism, files in organisms.items():
        # skip organisms that are currently empty
        if not files:
            continue
        # files can now be dict (QC) or dict of dicts (cell type)
        if isinstance(files, dict):
            for name, file in files.items():
                if not os.path.exists(ALL_PATHS["PATH_TO_DATABASE"] + file):
                    raise FileNotFoundError(
                        f'{file} does not exist in {ALL_PATHS["PATH_TO_DATABASE"]}'
                        ' please check!')
        else:
            raise TypeError(
                f"Unexpected structure in GENESET_FILENAMES[{k}][{organism}]")
# ########################################################################################
# For legacy function searching
def parse_python_file(
            file_path: str
        ) -> list[str] | list[dict[str, list[str]]]:
    """
    Parses a Python file and extracts the names of top-level functions and classes,
    including methods within classes.

    This function reads the Python source code from the specified file path and
    utilizes the Abstract Syntax Tree (AST) module to analyze the code
    structure. It collects the names of all top-level function definitions and
    classes, along with the methods within each class.

    NOTE:
        The function does not validate the syntax of the provided Python file
        beyond what is required for parsing.
        This is a helper for ``create_code_structure``

    Args:
        file_path (str): The path to the Python file to be parsed.

    Returns:
        list[str] | list[dict[str, list[str]]]:
            A list containing the names of top-level functions and
            dictionaries
            for classes where the class name is the key and the value is a
            list of method names.

    Called By:
        build_directory_structure

    TODO:
        Extend the function to handle nested functions and classes, and to
        optionally include function arguments in the output.

    Tags:
        codebase
    """
    # #########################################################
    # Reading the file content and parsing it into an AST
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    functions_and_classes = []
    # #########################################################
    # Iterating over top-level nodes in the AST to find functions and classes
    for node in tree.body:
        # ##########################################
        # Check if the node is a function definition and append its name
        if isinstance(node, ast.FunctionDef):
            functions_and_classes.append(node.name)
        # ##########################################
        # Check if the node is a class definition and extract its methods
        elif isinstance(node, ast.ClassDef):
            methods = [
                class_node.name for class_node in node.body
                if isinstance(class_node, ast.FunctionDef)]
            functions_and_classes.append({node.name: methods})
    # #########################################################
    # Returning the list of function names and class-method mappings
    return functions_and_classes


def build_directory_structure(
            abs_folder_path: str
        ) -> dict[str, list[str] | dict[str, list[str]]]:
    """
    Build a dictionary representing the directory structure of Python files
    within the specified root directory.

    This function traverses the directory tree rooted at ``abs_folder_path``,
    identifies all Python files (i.e., files with a ``.py`` extension), and
    generates a dictionary where each key is a Python filename, and each value
    is the result of parsing that file through the ``parse_python_file``
    function.

    NOTE:
        - This function assumes that the ``parse_python_file`` function is
          defined and available within the same module or is imported
          appropriately.
        - This is a helper for ``create_code_structure``

    Args:
        abs_folder_path (str): The root directory to start walking through to
            build the directory structure.

    Returns:
        dict: A dictionary where keys are Python filenames, and values are the
            parsed content of those files as processed by ``parse_python_file``.

    Raises:
        FileNotFoundError: If the provided ``abs_folder_path`` does not exist.
        TypeError: If ``abs_folder_path`` is not a string.

    Calls:
        parse_python_file

    Called By:
        create_code_structure

    TODO:
        Enhance this function to allow filtering for different file extensions
        beyond just ``.py``.

    Tags:
        codebase
    """
    # #########################################################
    # Initialize the dictionary to hold the directory structure
    structure = {}
    # #########################################################
    # Validate the input arguments
    if not isinstance(abs_folder_path, str):
        raise TypeError('abs_folder_path must be a string')
    if not os.path.exists(abs_folder_path):
        raise FileNotFoundError(f'The directory {abs_folder_path} does not exist')
    # #########################################################
    # Walk through the directory tree and process each Python file
    for dirpath, dirnames, filenames in os.walk(abs_folder_path):
        # ##########################################
        # Iterate over the filenames in the current directory
        for filename in filenames:
            # Only process files with a '.py' extension
            if filename.endswith('.py'):
                # ##########################################
                # Generate the full file path and parse the Python file
                file_path = os.path.join(dirpath, filename)
                structure[filename] = parse_python_file(file_path)
    # #########################################################
    # Return the final directory structure dictionary
    return structure


def create_code_structure() -> bool:
    """Generates the code structure for a project repository.

    This function builds the directory structure for a project repository by
    invoking the build_directory_structure function with the path to the
    repository. It is designed to ensure that the necessary directory tree is
    created, providing an organized structure for the project's files and
    folders.

    NOTE:
        Ensure that the path_to_repo variable is correctly set before calling
        this function.

    Returns:
        bool: Returns True if the directory structure was successfully created,
            False otherwise.

    Raises:
        FileNotFoundError: If the path to the repository does not exist.
        PermissionError: If there is insufficient permission to create
            directories in the specified path.

    Calls:
        build_directory_structure

    Called By:
        search_scToolkit_function_legacy

    Tags:
        codebase
    """
    # #########################################################
    # Build the directory structure using the path to the repository
    # NOTE: The path_to_repo variable should be defined and correctly set before this function is invoked.
    return build_directory_structure(path_to_repo)


# #########################################################
# Future database extractors
def get_hgnc(
            input_path: str,
            output_path: str = "",
            sep: str = "\t",
            key: str = "symbol",
            groupby: str = "locus_group",
        ) -> pd.DataFrame:
    """
    Load HGNC gene info, one column symbols per groupby group, and save to CSV.

    Args:
        input_path (str): Path to HGNC input file (CSV/TSV).
        output_path (str, optional): Path to save formatted CSV if set.
            Defaults to "".
        sep (str, optional): Delimiter used in the input file.
            Defaults to ``"\\t"``.
        key (str, optional): Column to use as gene key (e.g. ``"symbol"``).
            Defaults to ``"symbol"``.
        groupby (str, optional): Column to group on (e.g. ``"locus_group"``).
            Defaults to ``"locus_group"``.

    Returns:
        pd.DataFrame:
            Formatted dataframe with locus_group expanded.
    """
    # DOWNLOADED FROM: https://www.genenames.org/download/statistics-and-files/
    #                  Total Approved Symbols - the txt file
    # LINK: https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/non_alt_loci_set.txt
    # load_and_format_hgnc(ALL_PATHS["PATH_TO_DATABASE"] + "non_alt_loci_set.zip").to_csv(ALL_PATHS["PATH_TO_DATABASE"] + "HGNC_geneset_categories.csv", index=False)
    df = pd.read_csv(input_path, sep=sep)

    if groupby not in df.columns or key not in df.columns:
        raise ValueError(f"Missing required columns: {groupby}, {key}")

    # Normalize column names: replace spaces and dashes
    groups = (
        df.groupby(groupby)[key]
        .apply(list)
        .to_dict())
    groups = {
        str(k).replace(" ", "_").replace("-", "_"): v
        for k, v in groups.items()}

    # Pad to same length
    max_len = max(len(v) for v in groups.values())
    padded = {k: v + [""] * (max_len - len(v)) for k, v in groups.items()}

    out = pd.DataFrame(padded)

    if output_path:
        out.to_csv(output_path, index=False)

    return out


def get_mgi(
            input_path: str,
            output_path: str = "",
            sep: str = "\t",
            key: str = "Marker Symbol",
            groupby: str = "Marker Type",
            filter_regex: list[re.Pattern] =  [r"\(", r"\)", r"\<", r"\>", "/"],
        ) -> pd.DataFrame:
    """Load MGI gene info, one column of symbols per groupby group, and save.

    Each unique value in ``groupby`` becomes its own column, containing all
    ``key`` values for that group. Columns are padded with empty strings so
    they align to the longest list.

    Args:
        input_path (str): Path to MGI input file (e.g. MRK_List2.rpt).
        output_path (str, optional): Path to save formatted CSV if set.
            Defaults to "".
        sep (str, optional): Delimiter used in the input file.
            Defaults to ``"\\t"``.
        key (str, optional): Column to use as gene key (e.g.
            ``"Marker Symbol"``). Defaults to ``"Marker Symbol"``.
        groupby (str, optional): Column to group on (e.g. ``"Marker Type"``).
            Defaults to ``"Marker Type"``.
        filter_regex (list[str], optional): List of regex patterns.
            If any pattern matches the gene symbol, that gene is removed.
            Defaults to [r"\\(", r"\\)", r"\\>", r"\\>", r"/"]..

    Returns:
        pd.DataFrame:
            Formatted dataframe with groupby expanded into columns.
    """
    # DOWNLOADED FROM https://www.informatics.jax.org/downloads/reports/index.html
    # 1. List of Mouse Genetic Markers (sorted alphabetically by marker symbol, tab-delimited)
    #    the file MRK_List2.rpt.gz (excluding withdrawn marker symbols)
    # LINK:
    #     https://www.informatics.jax.org/downloads/reports/MRK_List2.rpt.gz
    # ##########################################
    # Load the database
    df = pd.read_csv(
        input_path,
        sep=sep,
        engine="python",
        on_bad_lines="skip")
    # ##########################################
    # Check if groupby is valid
    if groupby not in df.columns or key not in df.columns:
        raise ValueError(f"Missing required columns: {groupby}, {key}")
    # ##########################################
    # Apply regex filtering if patterns provided
    if filter_regex:
        combined_pattern = "|".join(filter_regex)
        # Drop rows where the key matches any of the regex patterns
        mask = ~df[key].astype(str).str.contains(combined_pattern, regex=True, na=False)
        df = df[mask]
    # ##########################################
    # groupby gene identifiers
    groups = df.groupby(groupby)[key].apply(list).to_dict()
    groups = {
        str(k).replace(" ", "_").replace("-", "_").replace("/", "_"): v
        for k, v in groups.items()
    }
    # ##########################################
    # Pad the genesets to create a dataframe
    max_len = max(len(v) for v in groups.values())
    padded = {k: v + [""] * (max_len - len(v)) for k, v in groups.items()}
    out = pd.DataFrame(padded)
    # ##########################################
    # Sanitycheck, because they tend provide wierd symbols.
    pattern = re.compile(r"[^A-Za-z0-9-._]")
    matched = False
    for col in out.columns:
        # Only check string-like columns
        if out[col].dtype == "object":
            mask = out[col].astype(str).str.contains(pattern)
            if mask.any():
                matched = True
                print(f"Column '{col}' has non-char entries:")
                print(df.loc[mask, col].unique()[:2])
    if matched:
        print("Please rerun and adjust the filter_regex!")
    # ##########################################
    # OPT. save and return dataframe
    if output_path:
        out.to_csv(output_path, index=False)
    return out