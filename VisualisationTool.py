import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class VisualisationTool:
    @staticmethod
    def plot_pca_simple(pca,  # input sklearn pca object
                        pca_res,  # transformed values
                        color_by=None,  # input the target COLUMN, NOT just its name
                        show_legend=True,
                        plotted_components=(1, 2),  # select which PCs to plot
                        ):
        # get loadings
        loadings = pd.DataFrame(pca.components_.T, columns=np.arange(1, pca.n_components_ + 1), index=pca.feature_names_in_)
        # get explained variance
        explained_variance = pca.explained_variance_ratio_

        # initialize figure with subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # plot PCA results, colored by categorical column 'color_by' if it was passed to the function
        if color_by is not None:
            for uniqueclass in set(color_by):
                indices_current = np.where(color_by == uniqueclass)
                ax1.scatter(pca_res[indices_current, 0], pca_res[indices_current, 1], label=uniqueclass)
                if show_legend:
                    ax1.legend()
        else:
            ax1.scatter(pca_res[:, 0], pca_res[:, 1])

        # make a loadings plot
        ax2.scatter(loadings.iloc[:, plotted_components[0] - 1], loadings.iloc[:, plotted_components[1] - 1])
        ax1.set_title('PCA plot')
        ax2.set_title('PCA loadings')

        xlabel = "PC" + str(plotted_components[0]) + " (" + str(round(explained_variance[plotted_components[0]-1], 4)) + ")"
        ylabel = "PC" + str(plotted_components[1]) + " (" + str(round(explained_variance[plotted_components[1]-1], 4)) + ")"

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        # show plot window
        plt.show()
        return
