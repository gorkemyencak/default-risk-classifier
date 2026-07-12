import pandas as pd

class PreProcessingEngine:
    """  
    Preprocessing engine helping to clean raw data into processed format

    Key functions:
        - renaming unclear column names
        - converting column data types into desirable format
        - dropping duplicate rows
        - dropping redundant, non-informative columns
        - dropping highly linearly correlated features
        - standardizing missing values
    """
    def __init__(self) -> None:
        # attributes
        self._column_rename_map = {
            'DisbursalDate': 'DisbursementDate'
        }

        self._columns_to_drop = [
            'State_ID',             # fully missing feature
            'Employee_code_ID',     # nunique is less than equal to 1
            'Mobile Avl Flag',      # nunique is less than equal to 1
            'UniqueID',             # non-informative feature
            'VoterID Flag',         # non-informative feature
            'Sanctioned Amount'     # highly linearly correlated feature
        ]

        self._columns_to_convert_datetime = [
            'DisbursementDate'
        ]

        self._missing_values_to_standardize = {
            'Employment Type': [
                'Missing',
                'missing',
                '',
                ' '
            ]
        }
    
    # preprocessing engine
    def _rename_columns(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Renaming unclear column names into easily interpretable ones """
        df = df.copy()
        df_renamed = df.rename(columns = self._column_rename_map)

        return df_renamed
    
    def _convert_into_datetime(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Converting time columns into datetime format """
        df = df.copy()

        for col in self._columns_to_convert_datetime:
            
            # sanity check
            if col not in df.columns:
                continue
            
            # convert into datetime series
            df[col] = pd.to_datetime(
                df[col],
                unit = 'D',
                origin = '1899-12-30',
                errors = 'coerce'
            )
        
        return df
    
    def _standardize_missing_values(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """  
        Transforming missing values into a standardized format

        _standardize_missing_values function will handle both NaN and other string categorical values that is intended to represent the 
        missing information in the dataset into a desirable single string categorical value. Missing categorical value by itself might be
        informative, so we preserve it as an explicit category instead of deleting rows
        """
        df = df.copy()

        for col in self._missing_values_to_standardize:
            # sanity check
            if col not in df.columns:
                continue

            # replace missing with 'Unknown'
            df[col] = df[col].fillna('Unknown')

            df[col] = df[col].replace(
                to_replace = self._missing_values_to_standardize[col],
                value = 'Unknown'
            )
        
        return df
    
    def _drop_duplicate_rows(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Dropping fully dupliated rows, since fully duplicated rows do not provide additional information """
        df = df.copy()
        
        return (
            df
            .drop_duplicates()
            .reset_index(drop = True)
        )
    
    def _drop_noninformative_columns(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Dropping columns identified during EDA as non-informative or redundant """
        df = df.copy()

        # sanity check whether columns to drop exist in df.columns
        cols_to_drop = [
            col
            for col in self._columns_to_drop if col in df.columns
        ]

        df_dropped = df.drop(columns = cols_to_drop)

        return df_dropped
    
    # preprocessing pipeline
    def fit(
            self,
            df: pd.DataFrame
    ):
        """ Fitting preprocessing engine """
        return self
    
    def transform(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Preprocessing pipeline to apply all preprocessing steps defined in the PreProcessingEngine """
        df_preprocessed = df.copy()

        # step 1: rename columns
        df_preprocessed = self._rename_columns(df = df_preprocessed)

        # step 2: convert into datetime
        df_preprocessed = self._convert_into_datetime(df = df_preprocessed)

        # step 3: standardizing missing values
        df_preprocessed = self._standardize_missing_values(df = df_preprocessed)

        # step 4: remove duplicate rows
        df_preprocessed = self._drop_duplicate_rows(df = df_preprocessed)

        # step 5: drop non-informative columns
        df_preprocessed = self._drop_noninformative_columns(df = df_preprocessed)

        return df_preprocessed
    
    def fit_transform(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Fit and transform the dataframe using PreProcessingEngine """
        self.fit(df = df)
        return self.transform(df = df)