from data_formatting import *
from Scaling import *
from explore_PCA import *
from VisualisationTool import *

# Read in the wine dataset and append the classes
wine_data = data_formatting("Wine.xlsx")

# Do different kind of scaling
mean_cent_scal_wine_data = mean_cent_scal_df = mean_centering_scale(wine_data)
autoscaling_wine_data = autoscaling_df = autoscaling(wine_data)
pareto_scal_wine_data = pareto_scal_df = pareto_scaling(wine_data)

# Using the autoscaling data for PCA
pca = pca_plot(autoscaling_wine_data)

# Visualize PCA
tool = VisualisationTool()
tool.plot_pca_simple(pca, ...)




