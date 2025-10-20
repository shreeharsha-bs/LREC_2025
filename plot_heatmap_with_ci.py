#!/usr/bin/env python3
"""
Generate a heatmap with bootstrapped 95% CI intervals for longform results.
Shows mean scores with confidence intervals for each condition and dimension.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
RESULTS_DIR = Path("buckeye_longform_results")
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95

# Gender mapping
GENDER_DICT = {
    "s01":"f","s02":"f","s03":"m","s04":"f","s05":"f","s06":"m","s07":"f","s08":"f","s09":"f","s10":"m",
    "s11":"m","s12":"f","s13":"m","s14":"f","s15":"m","s16":"f","s17":"f","s18":"f","s19":"m","s20":"f",
    "s21":"f","s22":"m","s23":"m","s24":"m","s25":"f","s26":"f","s27":"f","s28":"m","s29":"m","s30":"m",
    "s31":"f","s32":"m","s33":"m","s34":"m","s35":"m","s36":"m","s37":"f","s38":"m","s39":"f","s40":"m"
}

# Condition mapping (in order)
CONDITIONS = [
    "Modal",
    "Breathy",
    "Creaky", 
    "End Creak"
]

# Dimension names (as they appear in the JSON summary fields)
# Maps to all possible dimensions across different task types
DIMENSIONS = [
    "Agency of Advice",           # therapy
    "Emotional Validation",       # therapy
    "Improvement vs Retreat",     # therapy
    "Actionability of Plan",      # career
    "Role Status",                # career
    "Stereotypical Nature of Role", # career
    "Leadership Endorsement",     # interview
    "Shortlist Decision",         # interview
    "Salary Offer Bias",          # interview
    "Heroic Agency",              # story
    "Person in Distress",         # story
    "Achievement vs Relational Arc", # story
]

def bootstrap_ci(data, n_bootstrap=N_BOOTSTRAP, ci=CONFIDENCE_LEVEL):
    """Calculate bootstrapped confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    alpha = 1 - ci
    lower = np.percentile(boot_means, alpha/2 * 100)
    upper = np.percentile(boot_means, (1 - alpha/2) * 100)
    mean = np.mean(data)
    
    return mean, lower, upper

def load_scores():
    """Load all scores from JSON files, split by gender."""
    # Dictionary to store scores by gender, condition and dimension
    scores = {
        'male': {cond: {dim: [] for dim in DIMENSIONS} for cond in CONDITIONS},
        'female': {cond: {dim: [] for dim in DIMENSIONS} for cond in CONDITIONS}
    }
    
    # Load all score files
    for score_file in sorted(RESULTS_DIR.glob("*_scores.json")):
        # Extract condition and speaker from filename (1-4 maps to CONDITIONS index)
        filename = score_file.stem
        parts = filename.split('_')
        condition_idx = int(parts[0]) - 1
        speaker_id = parts[1]  # e.g., "s01"
        
        if condition_idx >= len(CONDITIONS):
            continue
        
        # Determine gender
        if speaker_id not in GENDER_DICT:
            continue
        
        gender = 'male' if GENDER_DICT[speaker_id] == 'm' else 'female'
        condition = CONDITIONS[condition_idx]
        
        # Load scores
        with open(score_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract dimension scores
        if 'summary' in data:
            for dim in DIMENSIONS:
                if dim in data['summary']:
                    scores[gender][condition][dim].append(data['summary'][dim])
    
    return scores

def create_heatmap(scores):
    """Create heatmap with bootstrapped CI, split by gender."""
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10))
    
    for gender, ax in [('female', ax1), ('male', ax2)]:
        # Calculate means and CIs for this gender
        means = np.zeros((len(DIMENSIONS), len(CONDITIONS)))
        ci_lower = np.zeros((len(DIMENSIONS), len(CONDITIONS)))
        ci_upper = np.zeros((len(DIMENSIONS), len(CONDITIONS)))
        ci_width = np.zeros((len(DIMENSIONS), len(CONDITIONS)))
        p_values = np.ones((len(DIMENSIONS), len(CONDITIONS)))  # Track p-values
        
        for i, dim in enumerate(DIMENSIONS):
            modal_data = np.array(scores[gender][CONDITIONS[0]][dim])  # Modal is first condition
            
            for j, cond in enumerate(CONDITIONS):
                data = np.array(scores[gender][cond][dim])
                if len(data) > 0:
                    mean, lower, upper = bootstrap_ci(data)
                    means[i, j] = mean
                    ci_lower[i, j] = lower
                    ci_upper[i, j] = upper
                    ci_width[i, j] = upper - lower
                    
                    # Mann-Whitney U test against Modal (skip Modal itself)
                    if j > 0 and len(modal_data) > 0:
                        _, p_val = stats.mannwhitneyu(modal_data, data, alternative='two-sided')
                        p_values[i, j] = p_val
        
        # Create heatmap
        sns.heatmap(means, 
                    annot=False,
                    fmt='.2f',
                    cmap='RdYlGn',
                    vmin=1, vmax=5,
                    cbar_kws={'label': 'Mean Score'},
                    xticklabels=CONDITIONS,
                    yticklabels=DIMENSIONS if gender == 'female' else [],
                    ax=ax,
                    linewidths=0.5,
                    linecolor='white')
        
        # Add mean values and CI as text annotations
        # Also add bold borders for significant differences
        alpha = 0.05
        for i in range(len(DIMENSIONS)):
            for j in range(len(CONDITIONS)):
                mean_val = means[i, j]
                ci_low = ci_lower[i, j]
                ci_high = ci_upper[i, j]
                
                # Determine text color based on background
                text_color = 'white' if mean_val < 2.5 else 'black'
                
                # Add mean and CI
                ax.text(j + 0.5, i + 0.4, f'{mean_val:.2f}',
                       ha='center', va='center', 
                       fontsize=14, fontweight='bold',
                       color=text_color)
                
                ax.text(j + 0.5, i + 0.7, f'[{ci_low:.2f}, {ci_high:.2f}]',
                       ha='center', va='center',
                       fontsize=9, style='italic',
                       color=text_color, alpha=0.9)
                
                # Add bold border if significantly different from Modal (p < 0.05)
                if p_values[i, j] < alpha:
                    rect = plt.Rectangle((j, i), 1, 1, 
                                        fill=False, 
                                        edgecolor='black', 
                                        linewidth=4, 
                                        zorder=10)
                    ax.add_patch(rect)
        
        # Styling
        gender_title = 'Female' if gender == 'female' else 'Male'
        ax.set_title(gender_title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Voice Quality Condition' if gender == 'male' else '', fontsize=13, fontweight='bold')
        if gender == 'female':
            ax.set_ylabel('Evaluation Dimension', fontsize=13, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(CONDITIONS, rotation=0, ha='center', fontsize=11)
        if gender == 'female':
            ax.set_yticklabels(DIMENSIONS, rotation=0, fontsize=11)
    
    # Main title
    fig.suptitle('Therapeutic Response Quality: Mean Scores with 95% CI by Gender\n(Bootstrapped, N=10,000; Bold border = p < 0.05 vs. Modal)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Also save as PDF for publication quality
    output_pdf = 'longform_heatmap_with_ci.pdf'
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✓ PDF version saved to: {output_pdf}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)
    for gender in ['female', 'male']:
        print(f"\n{'='*60}")
        print(f"{gender.upper()}")
        print(f"{'='*60}")
        for i, dim in enumerate(DIMENSIONS):
            print(f"\n{dim}:")
            for j, cond in enumerate(CONDITIONS):
                n_samples = len(scores[gender][cond][dim])
                if n_samples > 0:
                    data = np.array(scores[gender][cond][dim])
                    mean, lower, upper = bootstrap_ci(data)
                    
                    # Calculate p-value
                    modal_data = np.array(scores[gender][CONDITIONS[0]][dim])
                    p_val = 1.0
                    if j > 0 and len(modal_data) > 0:
                        _, p_val = stats.mannwhitneyu(modal_data, data, alternative='two-sided')
                    
                    sig_marker = " ***" if p_val < 0.05 else ""
                    # print(f"  {cond:15s}: μ={mean:.2f}, "f"95% CI=[{lower:.2f}, {upper:.2f}], "f"width={upper-lower:.2f}, n={n_samples}", end="")
                    # if j > 0:  # Show p-value for non-Modal conditions
                    #    print(f", p={p_val:.4f}{sig_marker}", end="")
                    print()

if __name__ == "__main__":
    print("Loading scores from longform_results...")
    scores = load_scores()
    
    # Check if we have data
    total_scores = sum(len(scores[gender][cond][dim]) 
                      for gender in ['male', 'female']
                      for cond in CONDITIONS 
                      for dim in DIMENSIONS)
    print(f"✓ Loaded {total_scores} total scores")
    
    # Print breakdown by gender
    female_scores = sum(len(scores['female'][cond][dim]) 
                       for cond in CONDITIONS 
                       for dim in DIMENSIONS)
    male_scores = sum(len(scores['male'][cond][dim]) 
                     for cond in CONDITIONS 
                     for dim in DIMENSIONS)
    print(f"  Female: {female_scores} scores")
    print(f"  Male: {male_scores} scores")
    
    print(f"\nGenerating heatmap with {N_BOOTSTRAP:,} bootstrap samples...")
    create_heatmap(scores)
    
    print("\n✓ Complete!")
