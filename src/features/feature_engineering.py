import pandas as pd
import numpy as np

class FeatureEngineeringEngine:
    """
    FeatureEngineeringEngine generating temporal features for classification/clustering models

    This class assumes that deterministic raw data preprocessing has already been applied by PreProcessingEngine

    Key features:
        - date related temporal features
        - FICO rating related temporal features
        - behavioral temporal features
    """
    def __init__(self) -> None:
        # attributes
        self.reference_disbursement_date = None

        self.fico_rating_column = 'FICO Rating'

        self.fico_rating_ordinal_values = {
            'Bureau Excluded': 0,
            'Poor': 1,
            'Fair': 2,
            'Good': 3,
            'Very Good': 4,
            'Exceptional': 5
        }

    # helper functions
    def _check_columns(
            self,
            df: pd.DataFrame,
            columns: set
    ) -> bool:
        """ Check whether the list of columns exists in the provided dataframe """
        return columns.issubset(df.columns)
    
    def _divide_series(
            self,
            numerator: pd.Series,
            denominator: pd.Series
    ) -> pd.Series:
        """ Dividing two pandas series, returning NaN when the denominator is missing or NaN """
        # safe division
        division = np.where(
            (denominator != 0) & (denominator.notna()),
            numerator / denominator,
            np.nan
        )

        return pd.Series(
            division,
            index = numerator.index
        )
    
    def _assign_fico_rating(
            self,
            fico_score: float
    ) -> str:
        """
        Assign FICO score ratings

        FICO ratings:
            800-850: Exceptional
            740-799: Very Good
            670-739: Good
            580-669: Fair
            300-579: Poor
            < 300: Bureau Excluded
        """
        # safety check for a missing fico score
        if pd.isna(fico_score):
            return 'Unknown'

        if fico_score < 300:
            return 'Bureau Excluded'
        elif fico_score < 580:
            return 'Poor'
        elif fico_score < 670:
            return 'Fair'
        elif fico_score < 740:
            return 'Good'
        elif fico_score < 800:
            return 'Very Good'
        else:
            return 'Exceptional'
    
    ### temporal features
    # date features
    def _date_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating date-based temporal features from DisbursementDate """
        df = df.copy()

        # sanity check
        if 'DisbursementDate' not in df.columns:
            return df
        
        if self.reference_disbursement_date is None:
            self.reference_disbursement_date = df['DisbursementDate'].min()
        
        # temporal features
        df['DisbursementYear'] = df['DisbursementDate'].dt.year
        df['DisbursementQuarter'] = df['DisbursementDate'].dt.quarter
        df['DisbursementMonth'] = df['DisbursementDate'].dt.month

        df['DaysSinceDisbursement'] = (
            df['DisbursementDate'] - self.reference_disbursement_date
        ).dt.days

        return df
    
    # fico rating features
    def _fico_rating_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating FICO rating temporal features using FICO score ranges defined by Fair Isaac Corporation """
        df = df.copy()

        # sanity check
        if 'FICO Score' not in df.columns:
            return df
        
        # bool column representing Bureau Excluded customers
        df['IsBureauExcluded'] = np.where(
            df['FICO Score'] < 300,
            1,
            0
        )

        # a separate column storing customers with FICO scores above 300
        df['Legit FICO Scores'] = np.where(
            df['FICO Score'] >= 300,
            df['FICO Score'],
            np.nan
        )

        # assigning FICO score ratings
        df[self.fico_rating_column] = df['FICO Score'].apply(self._assign_fico_rating)

        # ordinal values based on FICO score ratings
        df['FICO Rating Ordinal'] = df[self.fico_rating_column].map(self.fico_rating_ordinal_values)

        return df
    
    # financial behavioral features
    def _balance_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """  
        Creating balance related temporal features

        For balance related features, negative current balances will not be overwritten, instead:
            - the original value will be kept
            - a flag will be created
            - a non-negative balance version will be created for utilization ratios        
        """
        df = df.copy()

        # sanity check
        if 'Current Balance Amount' not in df.columns:
            return df
        
        # flags
        df['HasNegativeCurrentBalance'] = np.where(
            df['Current Balance Amount'] < 0,
            1,
            0
        )

        df['HasZeroCurrentBalance'] = np.where(
            df['Current Balance Amount'] == 0,
            1,
            0
        )

        df['CurrentBalancePositiveAmount'] = np.where(
            df['Current Balance Amount'] >= 0,
            1,
            0
        )

        # temporal features
        if self._check_columns(
            df = df,
            columns = {
                'Instalment Amount',
                'Disbursed Amount'
            }
        ):
            df['InstalmentToDisbursedRatio'] = self._divide_series(
                numerator = df['Instalment Amount'],
                denominator = df['Disbursed Amount']
            )

        if self._check_columns(
            df = df,
            columns = {
                'CurrentBalancePositiveAmount',
                'Disbursed Amount'
            }
        ):
            df['BalanceToDisbursedRatio'] = self._divide_series(
                numerator = df['CurrentBalancePositiveAmount'],
                denominator = df['Disbursed Amount']
            )
        
        return df
    
    def _account_activity_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating account activity related temporal features """
        df = df.copy()

        # sanity check
        if 'Number of Active Accounts' not in df.columns:
            return df
        
        # flag
        df['HasNoActiveAccounts'] = np.where(
            df['Number of Active Accounts'] == 0,
            1,
            0
        )

        # temporal feature
        if self._check_columns(
            df = df,
            columns = {
                'Number of Active Accounts',
                'Number of Accounts'
            }
        ):
            df['ActiveAccountRatio'] = self._divide_series(
                numerator = df['Number of Active Accounts'],
                denominator = df['Number of Accounts']
            )
        
        return df
    
    def _overdue_and_delinquency_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating overdue and delinquency risk temporal features """
        df = df.copy()

        # temporal features
        if self._check_columns(
            df = df,
            columns = {
                'Number of Overdue Accounts',
                'Number of Accounts'
            }
        ):
            df['OverdueAccountRatio'] = self._divide_series(
                numerator = df['Number of Overdue Accounts'],
                denominator = df['Number of Accounts']
            )
        
        if self._check_columns(
            df = df,
            columns = {
                'Number of Accounts Opened Last 6 Months',
                'Number of Accounts'
            }
        ):
            df['RecentlyOpenedAccountRatio'] = self._divide_series(
                numerator = df['Number of Accounts Opened Last 6 Months'],
                denominator = df['Number of Accounts']
            )
        
        if self._check_columns(
            df = df,
            columns = {
                'Number of Delinquencies Last 6 Months',
                'Number of Accounts'
            }
        ):
            df['RecentlyDelinquencyAccountRatio'] = self._divide_series(
                numerator = df['Number of Delinquencies Last 6 Months'],
                denominator = df['Number of Accounts']
            )
        
        return df
    
    def _credit_history_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating creit-history related temporal features """
        df = df.copy()

        # sanity check
        if 'Average Account Age' not in df.columns:
            return df
        
        # age band categorical values
        df['CreditHistoryAgeBand'] = pd.cut(
            df['Average Account Age'],
            bins = [
                -np.inf,
                5,
                11,
                35,
                59,
                np.inf
            ],
            labels = [
                'Very Young',
                'Young',
                'Established',
                'Mature',
                'Seasoned'
            ]
        )

        # temporal features
        if self._check_columns(
            df = df,
            columns = {
                'Number of Accounts',
                'Average Account Age'
            }
        ):
            df['AccountsPerCreditHistoryMonth'] = self._divide_series(
                numerator = df['Number of Accounts'],
                denominator = df['Average Account Age']
            )
        
        if self._check_columns(
            df = df,
            columns = {
                'Number of Accounts Opened Last 6 Months',
                'Average Account Age'
            }
        ):
            df['RecentlyOpenedAccountsPerCreditHistoryMonth'] = self._divide_series(
                numerator = df['Number of Accounts Opened Last 6 Months'],
                denominator = df['Average Account Age']
            )
        
        return df
    
    def _credit_inquiry_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating credit-inquiry related temporal features """
        df = df.copy()

        # sanity check
        if 'Number of Inquiries' not in df.columns:
            return df
        
        # inquiry intensity categorical values
        df['InquiryIntensityBand'] = pd.cut(
            df['Number of Inquiries'],
            bins = [
                -np.inf,
                0,
                1,
                3,
                np.inf
            ],
            labels = [
                'No Inquiry',
                'Single Inquiry',
                'Moderate Inquiries',
                'High Inquiries'
            ]
        )

        # temporal features
        if self._check_columns(
            df = df,
            columns = {
                'Number of Inquiries',
                'Average Account Age'
            }
        ):
            df['InquiryPerCreditHistoryMonth'] = self._divide_series(
                numerator = df['Number of Inquiries'],
                denominator = df['Average Account Age']
            )
        
        if self._check_columns(
            df = df,
            columns = {
                'Number of Inquiries',
                'Number of Accounts'
            }
        ):
            df['InquiryToAccountRatio'] = self._divide_series(
                numerator = df['Number of Inquiries'],
                denominator = df['Number of Accounts']
            )
        
        return df
    
    def _loan_amount_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Creating loan amount related temporal features """
        df = df.copy()

        # sanity check
        if 'Loan To Value' not in df.columns:
            return df
        
        # LTV-based flags
        df['IsHighLTV'] = np.where(
            df['Loan To Value'] >= 80,
            1,
            0
        )

        df['IsVeryHighLTV'] = np.where(
            df['Loan To Value'] >= 90,
            1,
            0
        )

        # a separate column storing LTVs below 100
        df['Legit LTV'] = np.where(
            df['Loan To Value'] <= 100,
            df['Loan To Value'],
            np.nan
        )

        # temporal feature
        if self._check_columns(
            df = df,
            columns = {
                'Instalment Amount',
                'Disbursed Amount'
            }
        ):
            df['InstalmentToDisbursedAmount'] = self._divide_series(
                numerator = df['Instalment Amount'],
                denominator = df['Disbursed Amount']
            )
        
        return df
    
    def _behavioral_features(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """  
        Creating temporal features to represent customer financial behavior

        Behavioral features will be shaped with respect to:
            - balance features
            - account activity features
            - overdue and delinquency risk features
            - credit-history features
            - credit-inquiry features
            - loan amount features
        """
        df = df.copy()

        # temporal features -> balance features
        df = self._balance_features(df = df)

        # temporal features -> account activity features
        df = self._account_activity_features(df = df)

        # temporal features -> overdue and delinquency risk features
        df = self._overdue_and_delinquency_features(df = df)

        # temporal features -> credit-history features
        df = self._credit_history_features(df = df)

        # temporal features -> credit-inquiry features
        df = self._credit_inquiry_features(df = df)

        # temporal features -> loan amount features
        df = self._loan_amount_features(df = df)

        return df
    
    # feature engineering pipeline
    def fit(
            self,
            df: pd.DataFrame
    ):
        """ Fit feature engineering engine on the training data """
        if 'DisbursementDate' in df.columns:
            self.reference_disbursement_date = df['DisbursementDate'].min()
        
        return self
    
    def transform(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Feature engineering pipeline method """
        df_features = df.copy()

        # step 1: date related temporal features
        df_features = self._date_features(df = df_features)

        # step 2: FICO rating related temporal features
        df_features = self._fico_rating_features(df = df_features)

        # step 3: customer financial behavior features
        df_features = self._behavioral_features(df = df_features)

        return df_features
    
    def fit_transform(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Fit and transform the dataframe using feature engineering pipeline """
        self.fit(df = df)
        return self.transform(df = df)