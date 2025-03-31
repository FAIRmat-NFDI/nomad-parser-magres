import os
from glob import glob


def get_files(
    pattern: str,
    filepath: str,
    stripname: str = '',
    deeper: bool = True,
    file_range: int = 10,
) -> list:
    """Get files following the `pattern` with respect to the file `stripname` (usually this
    being the mainfile of the given parser) up to / down from the `filepath` (`deeper=True` going
    down, `deeper=False` up)

    Args:
        pattern (str): targeted pattern to be found
        filepath (str): filepath to start the search
        stripname (str, optional): name with respect to which do the search. Defaults to ''.
        deeper (bool, optional): boolean setting the path in the folders to scan (down or up). Defaults to down=True.

    Returns:
        list: List of found files.
    """
    for _ in range(file_range):
        filenames = glob(f'{os.path.dirname(filepath)}/{pattern}')
        pattern = os.path.join('**' if deeper else '..', pattern)
        if filenames:
            break

    if len(filenames) > 1:
        # filter files that match
        suffix = os.path.basename(filepath).strip(stripname)
        matches = [f for f in filenames if suffix in f]
        filenames = matches if matches else filenames

    filenames = [f for f in filenames if os.access(f, os.F_OK)]
    return filenames
