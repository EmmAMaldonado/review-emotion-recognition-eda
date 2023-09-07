#%%
# SEGUIR DESDE ACA
# YA TENGO LOS DATOS GUARDADOS EN `MELTED_DF_PAIRED`
# AHORA TENGO QUE REPRODUCIR EL CODIGO QUE TENIA ANTES PARA PODER HACER META ANALYSIS CORRESPOMDIENTE CON ESTA DATA PAIREADA
#VOY A TENER QUE CALCULAR EL STANDARD DEVIATION 

import pandas as pd
import os

# Read data from the Excel file
melted_df = pd.read_excel(".\data\processed\melted_df_excel_paired.xlsx", engine='openpyxl')

# Print the first few rows of the DataFrame
print(melted_df.head())

#%%
# Print the data types of the columns in the filtered_models dataframe
print(melted_df.dtypes)

#%%
# Convert necessary columns to numeric types
melted_df['N'] = pd.to_numeric(melted_df['N'], errors='coerce')
melted_df['accuracy_arousal'] = pd.to_numeric(melted_df['accuracy_arousal'], errors='coerce')
melted_df['accuracy_valence'] = pd.to_numeric(melted_df['accuracy_valence'], errors='coerce')

#%%
# Drop any rows with missing values
melted_df = melted_df.dropna(subset=['N', 'accuracy_arousal', 'accuracy_valence'])


#%%
# Group the data by paper_id and calculate the standard deviation for arousal and valence
grouped_sd = melted_df.groupby('paper_id').agg({'accuracy_arousal': 'std', 'accuracy_valence': 'std'}).reset_index()

#%%
# Rename the columns
grouped_sd.columns = ['paper_id', 'SD_arousal', 'SD_valence']

# Merge the grouped_sd DataFrame with the filtered_models DataFrame on the paper_id column
melted_df = melted_df.merge(grouped_sd, on='paper_id', how='left')


# %%

filtered_models = melted_df


#%%
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go

# Assuming 'filtered_models' is the DataFrame containing your data

# Calculate the effect size for each model
filtered_models['effect_size'] = filtered_models['accuracy_arousal'] - filtered_models['accuracy_valence']
filtered_models['variance'] = (1 / filtered_models['N']) + (1 / filtered_models['N'])

# Group models by paper_id and calculate the mean effect size and variance for each paper
grouped_models = filtered_models.groupby('paper_id').agg({'effect_size': 'mean', 'variance': 'mean'}).reset_index()

# Calculate the weights
grouped_models['weight'] = 1 / grouped_models['variance']

# Calculate the weighted mean effect size
weighted_mean_effect_size = np.sum(grouped_models['weight'] * grouped_models['effect_size']) / np.sum(grouped_models['weight'])

# Calculate the variance of the weighted mean effect size
variance_weighted_mean_effect_size = 1 / np.sum(grouped_models['weight'])

# Calculate the standard error and confidence interval of the weighted mean effect size
standard_error = np.sqrt(variance_weighted_mean_effect_size)
critical_value = stats.norm.ppf(0.975)  # For a 95% confidence interval
lower_bound = weighted_mean_effect_size - critical_value * standard_error
upper_bound = weighted_mean_effect_size + critical_value * standard_error

# Print the results
print("Weighted mean effect size:", weighted_mean_effect_size)
print("95% Confidence interval:", (lower_bound, upper_bound))

# Add standard error, lower bound, and upper bound to the grouped_models DataFrame
grouped_models['standard_error'] = np.sqrt(grouped_models['variance'])
grouped_models['lower_bound'] = grouped_models['effect_size'] - critical_value * grouped_models['standard_error']
grouped_models['upper_bound'] = grouped_models['effect_size'] + critical_value * grouped_models['standard_error']

#%%
import statsmodels.api as sm

# Assuming you have effect sizes and variances as numpy arrays
effect_sizes = filtered_models['effect_size']
variances = np.array([...])

# Create a fixed-effects meta-analysis model
fixed_effects_model = sm.stats.DescrStatsW(effect_sizes, weights=1/variances)

# Calculate the Q statistic and its p-value
Q_statistic = fixed_effects_model.ttest_mean().statistic
Q_pvalue = fixed_effects_model.ttest_mean().pvalue

# Calculate the I² statistic
total_variance = np.var(effect_sizes, ddof=1)
sampling_error_variance = np.sum(variances) / len(variances)
heterogeneity_variance = max(total_variance - sampling_error_variance, 0)
I2 = (heterogeneity_variance / total_variance) * 100

print("Q statistic =", Q_statistic)
print("Q p-value =", Q_pvalue)
print("I² statistic =", I2)


#%%

papers_citations = [
    "Chang et al. (2019)",
    "Ganapathy & Swaminathan (2019)",
    "Siddharth et al. (2018)",
    " Ayata et al. (2017)",
    "Susanto et al. (2020)",
   "Ayata et al. (2017)",
    "Yin et al. (2019)",
    "Santamaria-Granados et al. (2018)",
    "Ganapathy & Swaminathan (2020)",
    "Wiem & Lachiri (2017)",
    "Sharma et al. (2019)",
    "Ganapathy et  (2020)"
]

# Sort grouped_models by 'effect_size' in ascending order
grouped_models = grouped_models.sort_values(by='effect_size', ascending=True)

# Reset index after sorting to rearrange paper_id order in plot
grouped_models.reset_index(drop=True, inplace=True)

# Create a scatter plot for individual study effect sizes and confidence intervals
fig = go.Figure()

for index, row in grouped_models.iterrows():
    citation = papers_citations[index]  # get corresponding citation
    fig.add_trace(
        go.Scatter(
            x=[row["lower_bound"], row["upper_bound"]],
            y=[citation, citation],  # replace index with citation
            mode="lines",
            showlegend=False,
            line=dict(color="#FF6961"),  # pastel red color for whiskers
            name="Confidence Interval"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row["effect_size"]],
            y=[citation],  # replace index with citation
            mode="markers",
            marker=dict(color="#FF6961"),  # pastel red color for mean (dot)
            showlegend=False,
            name="Effect Size"
        )
    )

fig.add_shape(
    type="line",
    x0=weighted_mean_effect_size,
    x1=weighted_mean_effect_size,
    y0=-1,
    y1=len(grouped_models),
    yref="y",
    line=dict(color="red"),
    name="Overall Effect Size"
)

fig.update_layout(
    xaxis_title="Effect Size (Arousal - Valence)",
    yaxis_title="Citation",
    yaxis=dict(autorange="reversed"),
    shapes=[dict(type='line', x0=0, x1=0, y0=-1, y1=len(grouped_models), yref="y", line=dict(color="black", dash='dash'), name="Line of No Effect")],
    legend=dict(
        x=1,
        y=1,
        traceorder="normal",
        font=dict(family="sans-serif", size=12),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=1
    )
)

fig.show()
# %%
import pandas as pd
from scipy.stats import wilcoxon

# Assuming filtered_models DataFrame is already loaded

# Extract accuracy_valence and accuracy_arousal columns
accuracy_valence = filtered_models['accuracy_valence']
accuracy_arousal = filtered_models['accuracy_arousal']

# Perform the Wilcoxon signed-rank test (one-sided)
w_stat, p_value = wilcoxon(accuracy_valence, accuracy_arousal, alternative='less')

# Print the test statistic and p-value
print(f"Wilcoxon signed-rank test statistic: {w_stat}")
print(f"Wilcoxon signed-rank test p-value: {p_value}")


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming filtered_models DataFrame is already loaded

# Set the plot style
sns.set(style="whitegrid", palette="pastel")

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Create the boxplot
sns.boxplot(data=filtered_models[['accuracy_valence', 'accuracy_arousal']], ax=ax, width=0.3)

# Create the scatterplot
sns.stripplot(data=filtered_models[['accuracy_valence', 'accuracy_arousal']], ax=ax, jitter=False, edgecolor="gray", alpha=0.6)

# Draw lines between points from each row
for index, row in filtered_models.iterrows():
    ax.plot([0, 1], [row['accuracy_valence'], row['accuracy_arousal']], color='gray', alpha=0.4)

# Set the title and labels
ax.set_title("Accuracy Valence and Accuracy Arousal")
ax.set_ylabel("Accuracy")
ax.set_xticklabels(['Accuracy Valence', 'Accuracy Arousal'])

# Show the plot
plt.show()
# %%
import pandas as pd
from scipy.stats import wilcoxon

# Assuming filtered_models DataFrame is already loaded

# Extract accuracy_valence and accuracy_arousal columns
accuracy_valence = filtered_models['accuracy_valence']
accuracy_arousal = filtered_models['accuracy_arousal']

# Perform the Wilcoxon signed-rank test (one-sided)
t, p_value = ttest(accuracy_valence, accuracy_arousal, alternative='less')

# Print the test statistic and p-value
print(f"Wilcoxon signed-rank test statistic: {t}")
print(f"Wilcoxon signed-rank test p-value: {p_value}")

# %%
