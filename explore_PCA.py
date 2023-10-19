
from sklearn.decomposition import PCA

import pandas as pd

def pca_plot(df):
    """
    Performs PCA on a given dataframe and returns the PCA object, 
    transformed data, and classes of the samples

    Parameters
    ----------
    df : pandas dataframe
        Pandas datafrae containing the scaled data of interest

    Returns
    -------
    pca : sklearn.decomposition.PCA
        PCA object of the inputted dataframe
        
    reduced_data: pandas dataframe
        transformed PCA data
        
    classes: series
        classes of the samples in the df
        
        
    """
    classes = df["Classes"]
    df = df.drop("Classes", axis=1)
    pca = PCA()
    pca.fit(df)
    reduced_data = pd.DataFrame(pca.transform(df))

    return pca, reduced_data, classes