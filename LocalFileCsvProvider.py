from typing import List

from CsvProvider import CsvProvider
import pandas as pd

class LocalFileCsvProvider(CsvProvider):
    def read_csv(self, path: str, delimiter: str, column_names: List[str]) -> pd.DataFrame:
        return pd.read_csv(path, delimiter=delimiter, names=column_names)