import pandas as pd
import matplotlib.pyplot as plt

def generate_pair_plot(csv_path):
    # 1. Load the dataset
    df = pd.read_csv(csv_path)
    
    # 2. Identify the target column and the features
    target_col = 'Hogwarts House'
    
    # DSLR dataset has an 'Index' column and some categorical columns to ignore.
    # We only want to plot the continuous numerical features (course grades).
    # You might want to manually specify the features to avoid clutter, 
    # but here we automatically select all float64 columns.
    features = df.select_dtypes(include=['float64']).columns.tolist()
    
    # Drop rows where the target column is missing
    df = df.dropna(subset=[target_col])
    
    # 3. Setup categories and colors
    categories = df[target_col].unique()
    colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'gold'
    }
    
    n = len(features)
    
    # 4. Create the matplotlib figure and axes grid
    fig, axes = plt.subplots(n, n, figsize=(17, 9))

    # 5. Populate the grid
    for i, y_feat in enumerate(features):
        for j, x_feat in enumerate(features):
            ax = axes[i, j]

            if i == j:
                # DIAGONAL: Only drop NaNs for the single feature being plotted
                plot_df = df[[x_feat, target_col]].dropna()
                
                for cat in categories:
                    subset = plot_df[plot_df[target_col] == cat]
                    # subset[x_feat] is now guaranteed to be a single Series
                    ax.hist(subset[x_feat], bins=15, alpha=0.5, color=colors.get(cat, 'gray'), label=cat)
            else:
                # OFF-DIAGONAL: Drop NaNs for both features to ensure x and y align
                plot_df = df[[x_feat, y_feat, target_col]].dropna()
                
                for cat in categories:
                    subset = plot_df[plot_df[target_col] == cat]
                    ax.scatter(subset[x_feat], subset[y_feat], alpha=0.6, s=5, color=colors.get(cat, 'gray'))
            
            # Clean up the axes to make the matrix readable
            if i < n - 1:
                ax.set_xticklabels([]) # Hide x-ticks for inner plots
            else:
                ax.set_xlabel(x_feat.replace(' ', '\n'), fontsize=8)
                
            if j > 0:
                ax.set_yticklabels([]) # Hide y-ticks for inner plots
            else:
                ax.set_ylabel(y_feat.replace(' ', '\n'), fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_pair_plot("data/dataset_train.csv")