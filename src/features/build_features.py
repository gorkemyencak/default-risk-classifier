import pandas as pd

from src.data.data_loader import (
    load_supervised_data,
    load_unsupervised_data,
    save_supervised_features,
    save_unsupervised_features
)

from src.preprocessing.data_preprocessing import PreProcessingEngine

from src.features.feature_engineering import FeatureEngineeringEngine

class FeatureBuilder:
    """  
    Build processed feature datasets from raw supervised and unsupervised data

    FeatureBuilder class will operate the full feature-building workflow:
        - load raw data
        - apply deterministic preprocessing
        - apply future engineering
        - optionally drop raw datetime columns after extracting date features
        - save processed feature datasets    
    """
    def __init__(
            self,
            preprocessor: PreProcessingEngine,
            feature_engineer: FeatureEngineeringEngine,
            drop_datetime_columns: bool = True
    ) -> None:
        # attributes
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.drop_datetime_columns = drop_datetime_columns
    
    # helper function
    def _drop_raw_datetime_columns(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Drop raw datetime columns after date related features have been created """
        df = df.copy()

        # extract datetime columns
        datetime_cols = df.select_dtypes(
            include = [
                'datetime64[ns]',
                'datetimetz'
            ]
        ).columns.tolist()

        if datetime_cols:
            df = df.drop(columns = datetime_cols)
        
        return df
    
    # feature builder engine
    def build_features_from_dataframe(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Build processed features from a given raw dataframe """
        df_processed = df.copy()

        # preprocessing
        df_processed = self.preprocessor.fit_transform(df = df_processed)

        # feature engineering
        df_processed = self.feature_engineer.fit_transform(df = df_processed)

        # drop raw datetime columns
        if self.drop_datetime_columns:
            df_processed = self._drop_raw_datetime_columns(df = df_processed)
        
        return df_processed
    
    def build_supervised_features(
            self,
            save: bool = True
    ) -> pd.DataFrame:
        """ Build processed supervised feature dataset """
        # load raw dataset
        supervised_raw = load_supervised_data()

        # build supervised features
        supervised_features = self.build_features_from_dataframe(df = supervised_raw)

        # save supervised features into the destination file path
        if save:
            save_supervised_features(df = supervised_features)
        
        return supervised_features
    
    def build_unsupervised_features(
            self,
            save: bool = True
    ) -> pd.DataFrame:
        """ Build processed unsupervised feature dataset """
        # load raw dataset
        unsupervised_raw = load_unsupervised_data()

        # build unsupervised features
        unsupervised_features = self.build_features_from_dataframe(df = unsupervised_raw)

        # save unsupervised features into the destination file path
        if save:
            save_unsupervised_features(df = unsupervised_features)
        
        return unsupervised_features
    
    def build_all_features(
            self,
            save: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Build both supervised and unsupervised feature datasets """
        # build supervised features
        supervised_features = self.build_supervised_features(save = save)

        # build unsupervised features
        unsupervised_features = self.build_unsupervised_features(save = save)

        return supervised_features, unsupervised_features