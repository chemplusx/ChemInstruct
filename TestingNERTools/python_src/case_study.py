import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_and_preprocess_data(data):
    # Convert string lists to actual lists
    df = pd.DataFrame(data)
    
    # Function to split and clean entities/interactions
    def clean_list(x):
        if isinstance(x, str):
            return [item.strip() for item in x.split('\n')]
        return []

    # Apply cleaning to relevant columns
    df['Chemical Entities'] = df['Chemical Entities'].apply(clean_list)
    df['Chemical/Drug Interactions'] = df['Chemical/Drug Interactions'].apply(clean_list)
    df['LLM Extracted Chemical Entities'] = df['LLM Extracted Chemical Entities'].apply(clean_list)
    df['LLM Evaluated Interaction - Chemical/Drug Interaction'] = df['LLM Evaluated Interaction - Chemical/Drug Interaction'].apply(clean_list)
    
    return df

def calculate_metrics(df):
    # Entity Recognition Metrics
    total_true_entities = sum(len(entities) for entities in df['Chemical Entities'])
    total_pred_entities = sum(len(entities) for entities in df['LLM Extracted Chemical Entities'])
    
    # Flatten all entities for comparison
    true_entities = set([entity.lower() for entities in df['Chemical Entities'] for entity in entities])
    pred_entities = set([entity.lower() for entities in df['LLM Extracted Chemical Entities'] for entity in entities])
    
    # Calculate overlap
    overlap = len(true_entities.intersection(pred_entities))
    
    # Entity metrics
    entity_precision = overlap / len(pred_entities)
    entity_recall = overlap / len(true_entities)
    entity_f1 = 2 * (entity_precision * entity_recall) / (entity_precision + entity_recall)
    
    # Interaction Metrics

    def to_lower_nested(value):
        if isinstance(value, list):
            return [to_lower_nested(v) for v in value]  # Recursively apply to elements
        return value.lower() if isinstance(value, str) else value
    # Convert all entries to lowercase for comparison
    formatted_data =df.applymap(to_lower_nested)
    true_interactions = [set(interactions) for interactions in formatted_data['Chemical/Drug Interactions']]
    pred_interactions = [set(interactions) for interactions in formatted_data['LLM Evaluated Interaction - Chemical/Drug Interaction']]
    
    # Calculate interaction metrics per document
    precisions, recalls, f1s = [], [], []
    for true, pred in zip(true_interactions, pred_interactions):
        if len(pred) == 0 and len(true) == 0:
            continue
        if len(pred) == 0:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
            continue
        if len(true) == 0:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
            continue
            
        overlap = len(true.intersection(pred))
        precision = overlap / len(pred)
        recall = overlap / len(true)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(precision, recall, f1, overlap, len(true), len(pred))
    
    metrics = {
        'Entity Recognition': {
            'Precision': entity_precision,
            'Recall': entity_recall,
            'F1': entity_f1,
            'Total True Entities': len(true_entities),
            'Total Predicted Entities': len(pred_entities),
            'Overlap': overlap
        },
        'Interaction Extraction': {
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'F1': np.mean(f1s)
        }
    }
    
    return metrics

def create_visualizations(df, metrics):
    # Set style
    plt.style.use('seaborn-v0_8-bright')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Metrics Comparison
    ax1 = plt.subplot(221)
    metrics_df = pd.DataFrame({
        'Entity Recognition': [metrics['Entity Recognition']['Precision'], 
                             metrics['Entity Recognition']['Recall'], 
                             metrics['Entity Recognition']['F1']],
        'Interaction Extraction': [metrics['Interaction Extraction']['Precision'],
                                 metrics['Interaction Extraction']['Recall'],
                                 metrics['Interaction Extraction']['F1']]
    }, index=['Precision', 'Recall', 'F1'])
    
    metrics_df.plot(kind='bar', ax=ax1)
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_ylim(0, 1)
    
    # 2. Entity Distribution
    ax2 = plt.subplot(222)
    entity_counts = pd.Series([len(entities) for entities in df['Chemical Entities']])
    predicted_entity_counts = pd.Series([len(entities) for entities in df['LLM Extracted Chemical Entities']])
    
    ax2.hist([entity_counts, predicted_entity_counts], label=['True Entities', 'Predicted Entities'], 
             alpha=0.7, bins=range(0, max(entity_counts.max(), predicted_entity_counts.max()) + 2))
    ax2.set_title('Distribution of Entities per Document')
    ax2.set_xlabel('Number of Entities')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Interaction Distribution
    ax3 = plt.subplot(223)
    interaction_counts = pd.Series([len(interactions) for interactions in df['Chemical/Drug Interactions']])
    predicted_interaction_counts = pd.Series([len(interactions) for interactions in df['LLM Evaluated Interaction - Chemical/Drug Interaction']])
    
    ax3.hist([interaction_counts, predicted_interaction_counts], 
             label=['True Interactions', 'Predicted Interactions'],
             alpha=0.7, bins=range(0, max(interaction_counts.max(), predicted_interaction_counts.max()) + 2))
    ax3.set_title('Distribution of Interactions per Document')
    ax3.set_xlabel('Number of Interactions')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 4. Entity Type Distribution
    ax4 = plt.subplot(224)
    # Analyze entity types (simplified)
    def get_entity_types(entities):
        types = []
        for entity in entities:
            if any(term in entity.lower() for term in ['cyp', 'p450']):
                types.append('Enzyme')
            elif any(term in entity.lower() for term in ['receptor', 'transporter']):
                types.append('Receptor/Transporter')
            else:
                types.append('Drug/Chemical')
        return types
    
    all_true_types = []
    for entities in df['Chemical Entities']:
        all_true_types.extend(get_entity_types(entities))
    
    type_counts = Counter(all_true_types)
    ax4.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
    ax4.set_title('Distribution of Entity Types')
    
    plt.tight_layout()
    return fig

def create_improved_visualizations(df, metrics):
    # Set style and color palette
    plt.style.use('seaborn-v0_8-bright')
    colors = {
        'entity': '#2E86C1',  # Professional blue
        'interaction': '#28B463',  # Complementary green
        'true': '#3498DB',  # Light blue
        'predicted': '#2ECC71'  # Light green
    }
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Metrics Comparison - Improved colors and formatting
    ax1 = plt.subplot(221)
    metrics_df = pd.DataFrame({
        'Entity Recognition': [metrics['Entity Recognition']['Precision'], 
                             metrics['Entity Recognition']['Recall'], 
                             metrics['Entity Recognition']['F1']],
        'Interaction Extraction': [metrics['Interaction Extraction']['Precision'],
                                 metrics['Interaction Extraction']['Recall'],
                                 metrics['Interaction Extraction']['F1']]
    }, index=['Precision', 'Recall', 'F1'])
    
    bar_width = 0.35
    x = np.arange(len(metrics_df.index))
    
    ax1.bar(x - bar_width/2, metrics_df['Entity Recognition'], 
            bar_width, label='Entity Recognition', color=colors['entity'])
    ax1.bar(x + bar_width/2, metrics_df['Interaction Extraction'],
            bar_width, label='Interaction Extraction', color=colors['interaction'])
    
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df.index)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 2. Entity Count Difference Analysis
    ax2 = plt.subplot(222)
    
    # Calculate differences between true and predicted entities
    true_counts = pd.Series([len(entities) for entities in df['Chemical Entities']])
    pred_counts = pd.Series([len(entities) for entities in df['LLM Extracted Chemical Entities']])
    differences = pred_counts - true_counts
    
    sns.histplot(differences, ax=ax2, bins=np.arange(min(differences)-0.5, max(differences)+1.5, 1),
                color=colors['entity'])
    ax2.set_title('Entity Count Differences\n(Predicted - True)', pad=20)
    ax2.set_xlabel('Difference in Number of Entities')
    ax2.set_ylabel('Number of Documents')
    
    # 3. Interaction Count Difference Analysis
    ax3 = plt.subplot(223)
    
    # Calculate differences between true and predicted interactions
    true_inter_counts = pd.Series([len(inter) for inter in df['Chemical/Drug Interactions']])
    pred_inter_counts = pd.Series([len(inter) for inter in df['LLM Evaluated Interaction - Chemical/Drug Interaction']])
    inter_differences = pred_inter_counts - true_inter_counts
    
    sns.histplot(inter_differences, ax=ax3, bins=np.arange(min(inter_differences)-0.5, max(inter_differences)+1.5, 1),
                color=colors['interaction'])
    ax3.set_title('Interaction Count Differences\n(Predicted - True)', pad=20)
    ax3.set_xlabel('Difference in Number of Interactions')
    ax3.set_ylabel('Number of Documents')
    
    # 4. Entity Type Distribution (Improved colors)
    ax4 = plt.subplot(224)
    def get_entity_types(entities):
        types = []
        for entity in entities:
            if any(term in entity.lower() for term in ['cyp', 'p450']):
                types.append('Enzyme')
            elif any(term in entity.lower() for term in ['receptor', 'transporter']):
                types.append('Receptor/Transporter')
            else:
                types.append('Drug/Chemical')
        return types
    
    all_true_types = []
    for entities in df['Chemical Entities']:
        all_true_types.extend(get_entity_types(entities))
    
    type_counts = Counter(all_true_types)
    colors_pie = ['#3498DB', '#2ECC71', '#E74C3C']  # Blue, Green, Red
    
    wedges, texts, autotexts = ax4.pie(type_counts.values(), 
                                      labels=type_counts.keys(),
                                      colors=colors_pie,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85)
    ax4.set_title('Distribution of Entity Types', pad=20)
    
    # Make percentage labels more readable
    plt.setp(autotexts, size=9, weight="bold")
    plt.setp(texts, size=9)
    
    plt.tight_layout()
    return fig

def generate_detailed_report(metrics):
    report = f"""
    Chemical NER and Interaction Extraction Analysis Report
    ====================================================
    
    1. Entity Recognition Performance
    -------------------------------
    - Precision: {metrics['Entity Recognition']['Precision']:.3f}
    - Recall: {metrics['Entity Recognition']['Recall']:.3f}
    - F1 Score: {metrics['Entity Recognition']['F1']:.3f}
    - Total True Entities: {metrics['Entity Recognition']['Total True Entities']}
    - Total Predicted Entities: {metrics['Entity Recognition']['Total Predicted Entities']}
    - Entity Overlap: {metrics['Entity Recognition']['Overlap']}
    
    2. Interaction Extraction Performance
    ----------------------------------
    - Precision: {metrics['Interaction Extraction']['Precision']:.3f}
    - Recall: {metrics['Interaction Extraction']['Recall']:.3f}
    - F1 Score: {metrics['Interaction Extraction']['F1']:.3f}
    """
    return report

# Example usage:
df = pd.read_csv('H:\\workspace\\Part1\\ChemInstruct\\TestingNERTools\\python_src\\Prompt Metrics - Case Study (1).csv')
df = load_and_preprocess_data(df)
metrics = calculate_metrics(df)
# fig = create_visualizations(df, metrics)
fig = create_improved_visualizations(df, metrics)
report = generate_detailed_report(metrics)

# Print report and show visualizations
print(report)
plt.show()