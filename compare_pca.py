import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

test_df = pd.DataFrame({'Classes': [0, 1, 2, 0, 1, 2],
        'Feature1': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4],
        'Feature2': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9]})

# USE: from compare_pca import two_class_pca
#  pca, X_new, labels = two_class_pca(wine_df)

def two_classes(wine_df, classes=[1,2]):
	labels = wine_df['Classes']
	# remove the label column
	data = wine_df.drop(columns=['Classes'])

	# select the samples in classes
	indices = labels[labels.isin(classes)].index

	# (Samples x Features)
	X = data.values
	X_sel = X[indices]
	
	# all PCs
	pca = PCA(centering=False)
	X_new = pca.fit_transform(X_sel)

	# convert series to string
	labels = labels.astype(str)
	return pca, X_new, labels