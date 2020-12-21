from typing import List

import pandas as pd

class CsvProvider:
    def read_csv(self, path: str, delimiter: str, column_names: List[str]) -> pd.DataFrame:
        pass