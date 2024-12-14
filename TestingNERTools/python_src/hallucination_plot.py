import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print(plt.style.available)

# Create DataFrame
data = {
    'Model': ['LLaMA-2 Chat FT', 'LLaMA-2 FT', 'Gemini-pro', 'Gpt-3.5-turbo-0125', 
              'Gpt-4-1106-preview', 'LLaMA-2 Chat', 'LLaMA-2 Chat with RAG', 
              'LLaMA-2 Chat FT with RAG', 'LLaMA-2 FT with RAG'],
    'HR': [0.24, 0.237, 0.244, 0.211, 0.215, 0.201, 0.04, 0.0201, 0.019],
    'CCS': [0.911, 0.905, 0.925, 0.922, 0.901, 0.935, 0.957, 0.956, 0.988],
    'CVS': [0.825, 0.851, 0.897, 0.914, 0.931, 0.914, 0.99, 0.989, 0.981],
    'Combined_Metric': [0.1752, 0.168, 0.151, 0.1336, 0.1364, 0.1257, 0.0319, 0.02454, 0.0169]
}

df = pd.DataFrame(data)

# Set style
plt.style.use('seaborn-v0_8-bright')
sns.set_palette("husl")

# Figure 1: Combined Performance Metrics
plt.figure(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.2

plt.bar(x - width*1.5, df['HR'], width, label='Hallucination Rate')
plt.bar(x - width/2, df['CCS'], width, label='Context Consistency Score')
plt.bar(x + width/2, df['CVS'], width, label='Chemical Validity Score')
plt.bar(x + width*1.5, df['Combined_Metric'], width, label='Combined Metric')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Performance Metrics Across Different Models')
plt.xticks(x, df['Model'], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, alpha=0.3)

# Figure 2: Performance Comparison Heatmap
plt.figure(figsize=(12, 8))
metrics_df = df.set_index('Model')[['HR', 'CCS', 'CVS', 'Combined_Metric']]
sns.heatmap(metrics_df, annot=True, cmap='YlOrRd_r', fmt='.3f', 
            cbar_kws={'label': 'Score'})
plt.title('Performance Metrics Heatmap')
plt.tight_layout()

# Figure 3: Model Comparison Radar Chart
def make_spider(df, title):
    metrics = ['HR', 'CCS', 'CVS', 'Combined_Metric']
    num_vars = len(metrics)
    
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], metrics)
    
    for idx, model in enumerate(df['Model']):
        values = df.loc[idx, metrics].values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(0.9, 0.9), loc='upper left', bbox_transform=plt.gcf().transFigure)
    plt.tight_layout()

make_spider(df, 'Model Performance Comparison - Radar Chart')

plt.show()