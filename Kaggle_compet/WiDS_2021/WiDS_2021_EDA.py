# Hàm đánh giá feature importance dựa trên cross entropy của các biến cateogry => biến binary 
def evaluate_binary_cross_entropy(df, target_col):
    '''Tính binary cross entropy của 1 biến categorical
    params:
        :df:dataframe: dataframe with target and categorical feature columns
        :target_col:str: target column name, with binary value
    return: dictionary of column name and cross entropy'''
    cols_ = [i for i in df.columns if i != target_col ]
    ce_list = {}
    for col in cols_:
        x = df[ [col, target_col] ].groupby(col).agg({target_col: lambda x: np.sum(x)/len(x), col:lambda x: len(x)/df.shape[0]})
        x.columns = ['p', 'q']
        ce = -sum([x.iloc[i,:]['p']*np.log2(x.iloc[i,:]['q']) for i in range(len(x))])
        ce_list[col] = ce 
    return ce_list

def evaluate_num_bce(df, target_col, bin = 100):
    '''evaluate binary cross entropy of the numeric data'''
    cols_ = [i for i in df.columns if i != target_col ]
    ce_list = {}
    df = df.copy()
    for col in cols_:
        df[col] = pd.cut(df[col], bin)
    return evaluate_binary_cross_entropy(df, target_col)


def _feature_generation(df):
    df = df.copy()
    # hospital_admit_source = other => nhãn 0
    df['hospital_admit_source_unknow'] = df['hospital_admit_source'].fillna('Other').apply(lambda x: 1 if x == 'Other' else 0)
    return df 

_feature_generation(df_train.head())


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.

    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target 
            encode all categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X 
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)