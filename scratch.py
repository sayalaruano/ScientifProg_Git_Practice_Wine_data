import pandas as pd
from sklearn.decomposition import PCA
from VisualisationTool import VisualisationTool
import numpy as np

colnames = ['feature1',
            'feature2',
            'feature3',
            'feature4',
            'feature5',
            'feature6',
            'feature7',
            'feature8',
            'feature9',
            'feature10',
            'feature11',
            'feature12',
            ]

data = pd.read_excel('Wine.xlsx', header=None, names=colnames)
print(data)

# initialize sklearn's PCA object
pca = PCA(n_components=3)
# load data and fit model
pca_res = pca.fit_transform(data)

color_by = np.ones(len(data))
color_by = color_by.tolist()

VisualisationTool.plot_pca_simple(pca=pca,
                                  pca_res=pca_res,
                                  color_by=color_by,
                                  show_legend=True,
                                  plotted_components=(1, 2),
                                  )


