"""
Exploratory Data Analysis module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import plot
import os
from config import PLOTS_DIR


def create_plots_directory():
    """Create plots directory if it doesn't exist"""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)


def plot_class_imbalance(data_df, save=True):
    """
    Plot class imbalance distribution
    
    Args:
        data_df (pd.DataFrame): Input dataset
        save (bool): Whether to save the plot
    """
    create_plots_directory()
    
    temp = data_df["Class"].value_counts()
    df = pd.DataFrame({'Class': temp.index, 'values': temp.values})
    
    trace = go.Bar(
        x=df['Class'], y=df['values'],
        name="Credit Card Fraud Class - data unbalance",
        marker=dict(color="Red"),
        text=df['values']
    )
    
    layout = dict(
        title='Credit Card Fraud Class Distribution',
        xaxis=dict(title='Class', showticklabels=True),
        yaxis=dict(title='Number of transactions'),
        hovermode='closest',
        width=600
    )
    
    fig = dict(data=[trace], layout=layout)
    
    if save:
        plot(fig, filename=f'{PLOTS_DIR}/class_imbalance.html', auto_open=False)
    
    return fig


def plot_time_distribution(data_df, save=True):
    """
    Plot transaction time density distribution
    
    Args:
        data_df (pd.DataFrame): Input dataset
        save (bool): Whether to save the plot
    """
    create_plots_directory()
    
    class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
    class_1 = data_df.loc[data_df['Class'] == 1]["Time"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(class_0, label='Not Fraud', fill=True, ax=ax)
    sns.kdeplot(class_1, label='Fraud', fill=True, ax=ax)
    
    ax.set_title('Credit Card Transactions Time Density Plot')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Density')
    ax.legend()
    
    if save:
        plt.savefig(f'{PLOTS_DIR}/time_distribution.png', dpi=100, bbox_inches='tight')
        print("  ✓ Time distribution plot saved")
    
    plt.close(fig)
    return fig


def plot_amount_distribution(data_df, save=True):
    """
    Plot transaction amount distribution
    
    Args:
        data_df (pd.DataFrame): Input dataset
        save (bool): Whether to save the plot
    """
    create_plots_directory()
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    sns.boxplot(ax=ax1, x="Class", y="Amount", hue="Class", data=data_df, 
                palette="PRGn", showfliers=True, legend=False)
    sns.boxplot(ax=ax2, x="Class", y="Amount", hue="Class", data=data_df, 
                palette="PRGn", showfliers=False, legend=False)
    
    if save:
        plt.savefig(f'{PLOTS_DIR}/amount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(data_df, save=True):
    """
    Plot feature correlation heatmap
    
    Args:
        data_df (pd.DataFrame): Input dataset
        save (bool): Whether to save the plot
    """
    create_plots_directory()
    
    plt.figure(figsize=(14, 14))
    plt.title('Credit Card Transactions features correlation plot (Pearson)')
    corr = data_df.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                linewidths=.1, cmap="Reds")
    
    if save:
        plt.savefig(f'{PLOTS_DIR}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_distributions(data_df, save=True):
    """
    Plot distributions of all features by class
    
    Args:
        data_df (pd.DataFrame): Input dataset
        save (bool): Whether to save the plot
    """
    create_plots_directory()
    
    var = data_df.columns.values
    t0 = data_df.loc[data_df['Class'] == 0]
    t1 = data_df.loc[data_df['Class'] == 1]
    
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(8, 4, figsize=(16, 28))
    
    for i, feature in enumerate(var):
        plt.subplot(8, 4, i + 1)
        sns.kdeplot(t0[feature], label="Class = 0", warn_singular=False)
        sns.kdeplot(t1[feature], label="Class = 1", warn_singular=False)
        plt.xlabel(feature, fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{PLOTS_DIR}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_all_eda_plots(data_df):
    """
    Generate all EDA plots
    
    Args:
        data_df (pd.DataFrame): Input dataset
    """
    print("Generating EDA plots...")
    
    # Add Hour feature
    data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))
    
    plot_class_imbalance(data_df)
    print("  ✓ Class imbalance plot saved")
    
    plot_time_distribution(data_df)
    print("  ✓ Time distribution plot saved")
    
    plot_amount_distribution(data_df)
    print("  ✓ Amount distribution plot saved")
    
    plot_correlation_matrix(data_df)
    print("  ✓ Correlation matrix saved")
    
    plot_feature_distributions(data_df)
    print("  ✓ Feature distributions saved")
    
    print(f"All plots saved to {PLOTS_DIR}/")
