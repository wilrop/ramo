import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)


def save_plot(plot_name, filetype, dpi=300):
    """Save a generated plot with the requested name and filetype.

    Args:
        plot_name (str): The file to save the plot to.
        filetype (str): The filetype for the figure.
        dpi (int, optional): The resolution for the final figure. Used when saving as png. (Default value = 300)

    Returns:

    """
    plt.tight_layout()
    if filetype == 'png':
        plt.savefig(plot_name + ".png", dpi=dpi)
    else:
        plt.savefig(plot_name + ".pdf")

    plt.clf()


def lineplot(df, x_col, y_col, label, x_label=None, y_label=None, x_min=None, x_max=None, y_min=None, y_max=None, linestyle='-', linewidth=2.0):
    """Draw a line plot.

    Args:
        df (DataFrame): A dataframe containing the data to plot.
        x_col (str): The column in the dataframe for the x-axis.
        y_col (str): The column in the dataframe for the y-axis.
        label (str): The label for the line.
        x_label (str, optional): The x-axis label. (Default value = None)
        y_label (str, optional):The y-axis label. (Default value = None)
        x_min (float, optional): The minimum value on the x-axis. (Default value = None)
        x_max (float, optional): The maximum value on the x-axis. (Default value = None)
        y_min (float, optional): The minimum value on the y-axis. (Default value = None)
        y_max (float, optional): The maximum value on the y-axis. (Default value = None)
        linestyle (str, optional): The linestyle for the line plot. (Default value = '-')
        linewidth (float, optional): The width of the line. (Default value = 2.0)

    Returns:
        Axes: The axes containing the line plot.
    """
    ax = sns.lineplot(x=x_col, y=y_col, linewidth=linewidth, data=df, ci='sd', label=label)
    ax = ax.set(xlabel=x_label, ylabel=y_label, xlim=(x_min, x_max), ylim=(y_min, y_max))
    return ax


def plot_constant(df, constant, x_col, label, y_label=None, x_label=None, x_min=None, x_max=None, y_min=None, y_max=None, linestyle='--', linewidth=1.0):
    """Draw a constant on the plot.

    Note:
        This can for example be useful to plot lower or upper bounds or to visualise a theoretical optimum.

    Args:
        df (DataFrame): A dataframe containing the data to plot.
        constant (float): The constant value for the y-axis.
        x_col (str): The column in the dataframe for the x-axis.
        label (str): The label for the line.
        x_label (str, optional): The x-axis label. (Default value = None)
        y_label (str, optional):The y-axis label. (Default value = None)
        x_min (float, optional): The minimum value on the x-axis. (Default value = None)
        x_max (float, optional): The maximum value on the x-axis. (Default value = None)
        y_min (float, optional): The minimum value on the y-axis. (Default value = None)
        y_max (float, optional): The maximum value on the y-axis. (Default value = None)
        linestyle (str, optional): The style for the line. (Default value = '-')
        linewidth (float, optional): The width of the line. (Default value = 2.0)

    Returns:
        Axes: The axes containing the line plot.
    """
    y_col = 'constant'
    df[y_col] = constant
    return lineplot(df, x_col, y_col, label, y_label=y_label, x_label=x_label, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, linestyle=linestyle, linewidth=linewidth)
