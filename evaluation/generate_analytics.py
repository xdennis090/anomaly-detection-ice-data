import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    
    thematic_mapping = {
        "09.08.2024": "Technologie", "07.09.2024": "Technologie",
        "14.11.2024": "Sozioökonomie", "30.06.2025": "Infrastruktur",
        "01.07.2025": "Meteorologie", "02.07.2025": "Meteorologie",
        "03.07.2025": "Meteorologie", "04.07.2025": "Meteorologie",
        "10.07.2025": "Infrastruktur", "14.07.2025": "Infrastruktur",
        "19.09.2025": "Infrastruktur", "22.09.2025": "Technologie",
        "13.10.2025": "Infrastruktur", "15.10.2025": "Technologie",
        "23.10.2025": "Meteorologie", "24.10.2025": "Meteorologie",
        "30.10.2025": "Infrastruktur", "12.01.2026": "Meteorologie",
        "26.01.2026": "Meteorologie", "27.01.2026": "Meteorologie",
        "30.01.2026": "Sozioökonomie"
    }
    df['Thematic_Cluster'] = df['Datum'].map(thematic_mapping)
    
    def get_temporal_cluster(date):
        if "2024" in date: 
            return "A (Initialphase 2024)"
        elif "2025" in date and (".06." in date or ".07." in date): 
            return "B (Sommer-Kollaps 2025)"
        elif "2025" in date and ".09." in date: 
            return "C1 (September-Plateau 2025)"
        elif "2025" in date and ".10." in date: 
            return "C2 (Oktober-Volatilität 2025)"
        elif "2026" in date: 
            return "D (Winterfenster 2026)"
        return "Unbekannt"
    
    df['Temporal_Cluster'] = df['Datum'].apply(get_temporal_cluster)
    return df

def plot_barchart(df, output_dir):
    metrics = ['Score_K1', 'Score_K2', 'Score_K3', 'Score_K4', 'Gesamt_Score']
    labels = ['K1', 'K2', 'K3', 'K4', 'Gesamtbewertung']
    
    means = df[metrics].mean()
    stds = df[metrics].std()
    
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color='#2b5c8f', edgecolor='black', alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        formatted_value = f'{height:.2f}'.replace('.', ',')
        
        plt.text(
            bar.get_x() + bar.get_width()/2.0 + 0.06, 
            height + 0.02,
            formatted_value, 
            ha='left', va='bottom', fontsize=10
        )
        
    plt.ylabel('Score')
    plt.ylim(0, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagramm_6_2_4.pdf'), bbox_inches='tight')
    plt.close()

def plot_boxplot(df, output_dir):
    plt.figure(figsize=(7.5, 4.5))
    df['Thematic_Cluster'] = df['Thematic_Cluster'].astype(str).str.strip()
    order = ['Technologie', 'Meteorologie', 'Sozioökonomie', 'Infrastruktur']
    
    sns.boxplot(x='Thematic_Cluster', y='Gesamt_Score', data=df, order=order, 
                palette='Blues_r', hue='Thematic_Cluster', legend=False, showfliers=False)
    sns.stripplot(x='Thematic_Cluster', y='Gesamt_Score', data=df, order=order, 
                  color='black', alpha=0.65, jitter=0.24, size=6, edgecolor='white', linewidth=0.6)
    
    plt.xlabel('Cluster')
    plt.ylabel('Score')
    plt.ylim(-0.2, 5.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagramm_6_3_5.pdf'), bbox_inches='tight')
    plt.close()

def plot_overview(df, output_dir):
    df_sorted = df.copy()
    df_sorted['Datetime'] = pd.to_datetime(df_sorted['Datum'], format='%d.%m.%Y')
    df_sorted = df_sorted.sort_values('Datetime').reset_index()
    
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(df_sorted['Datum'], df_sorted['Gesamt_Score'], marker='o', color='#d95f02', linewidth=2.5, label='Gesamtbewertung')
    
    global_mean = df_sorted['Gesamt_Score'].mean()
    mean_label = f'Mittelwert (µ = {global_mean:.2f})'.replace('.', ',')
    plt.axhline(y=global_mean, color='gray', linestyle='--', alpha=0.7, label=mean_label)
    
    plt.axvspan(-0.5, 2.5, color='#4a6572', alpha=0.25, label='2024')
    plt.axvspan(2.5, 9.5, color='#a64b4b', alpha=0.25, label='Sommer 2025')
    plt.axvspan(9.5, 11.5, color='#5d7a68', alpha=0.25, label='Spätsommer 2025')
    plt.axvspan(11.5, 16.5, color='#c97a3e', alpha=0.25, label='Herbst 2025')
    plt.axvspan(16.5, 20.5, color='#2b5c8f', alpha=0.25, label='Winter 2026')
    
    for x_boundary in [2.5, 9.5, 11.5, 16.5]:
        plt.axvline(x=x_boundary, color='#7f7f7f', linestyle=':', linewidth=1.2, alpha=0.6)
        
    plt.ylabel('Score')
    plt.xlabel('Untersuchungszeitraum')
    plt.xlim(-0.5, 20.5) 
    plt.ylim(-0.2, 5.5)
    
    plt.xticks(rotation=55, ha='right')
    
    plt.legend(
        loc='lower center', bbox_to_anchor=(0.5, 1.05), 
        ncol=3, borderaxespad=0, fontsize=10, frameon=True
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagramm_6_4_6.pdf'), bbox_inches='tight')
    plt.close()

def generate_evaluation_matrix(df, output_dir):
    df_sorted = df.copy()
    df_sorted['Datetime'] = pd.to_datetime(df_sorted['Datum'], format='%d.%m.%Y')
    df_sorted = df_sorted.sort_values('Datetime').reset_index(drop=True)
    df_sorted['Szenario_ID'] = df_sorted.index + 1
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Evaluationsmatrix der untersuchten Anomalien}\label{tab:master_matrix}")
    latex_lines.append(r"\footnotesize")
    latex_lines.append(r"\begin{tabularx}{\textwidth}{ccr>{\raggedright\arraybackslash}Xrrrrr}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"ID & Datum & Delay [Min.] & Störungsursache (Ground Truth) & K1 & K2 & K3 & K4 & Gesamt \\")
    latex_lines.append(r"\midrule")
    
    for _, row in df_sorted.iterrows():
        delay_fmt = f"{row['Delay_Min']:.2f}".replace('.', ',')
        k1_fmt = f"{row['Score_K1']:.2f}".replace('.', ',')
        k2_fmt = f"{row['Score_K2']:.2f}".replace('.', ',')
        k3_fmt = f"{row['Score_K3']:.2f}".replace('.', ',')
        k4_fmt = f"{row['Score_K4']:.2f}".replace('.', ',')
        total_fmt = f"{row['Gesamt_Score']:.2f}".replace('.', ',')
        
        clean_ground_truth = str(row['Ursache_Ground_Truth']).replace('&', r'\&')
        
        line = (f"{row['Szenario_ID']} & {row['Datum']} & {delay_fmt} & "
                f"{clean_ground_truth} & {k1_fmt} & {k2_fmt} & "
                f"{k3_fmt} & {k4_fmt} & \\textbf{{{total_fmt}}} \\\\")
        latex_lines.append(line)
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabularx}")
    latex_lines.append(r"\end{table}")
    
    output_path = os.path.join(output_dir, 'tabelle_2_evaluationsmatrix.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))


if __name__ == "__main__":
    csv_path = 'evaluation/results.csv'
    output_dir = 'evaluation'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Lade Daten aus der Master-CSV...")
    df = load_and_prepare_data(csv_path)
    
    print("Erstelle Balkendiagramm")
    plot_barchart(df, output_dir)
    
    print("Erstelle Boxplot")
    plot_boxplot(df, output_dir)
    
    print("Erstelle Gesamtübersicht")
    plot_overview(df, output_dir)
    
    print("Erstelle Evaluationsmatrix")
    generate_evaluation_matrix(df, output_dir)