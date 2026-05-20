import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_mixed_csv(csv_path):
    # Definisi header penuh yang menampung metrik validasi
    columns = ['Epoch', 'AvgLoss', 'AvgBCELoss', 'Precision', 'Recall', 'F1_Score', 'IoU', 'Dice',
               'ValLoss', 'ValBCELoss', 'ValPrecision', 'ValRecall', 'ValF1_Score', 'ValIoU', 'ValDice']
    
    data = []
    with open(csv_path, 'r') as f:
        # Lewati baris pertama (header lama)
        first_line = f.readline()
        
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(',')
            
            # Abaikan jika baris tersebut adalah header baru yang kita tulis di tengah jalan
            if parts[0] == 'Epoch':
                continue
            
            # Buat dictionary baru dengan default nilai NaN/None
            row = {col: None for col in columns}
            
            # Epoch
            try:
                row['Epoch'] = float(parts[0])
            except ValueError:
                continue # Jika tidak bisa diubah jadi angka, lewati baris ini
                
            # Parse metrik Train (selalu ada)
            row['AvgLoss'] = float(parts[1])
            row['AvgBCELoss'] = float(parts[2])
            row['Precision'] = float(parts[3])
            row['Recall'] = float(parts[4])
            row['F1_Score'] = float(parts[5])
            row['IoU'] = float(parts[6])
            row['Dice'] = float(parts[7])
            
            # Parse metrik Val jika sudah ada kolomnya di CSV (index 8 s/d 14)
            if len(parts) >= 15:
                row['ValLoss'] = float(parts[8])
                row['ValBCELoss'] = float(parts[9])
                row['ValPrecision'] = float(parts[10])
                row['ValRecall'] = float(parts[11])
                row['ValF1_Score'] = float(parts[12])
                row['ValIoU'] = float(parts[13])
                row['ValDice'] = float(parts[14])
                
            data.append(row)
            
    df = pd.DataFrame(data)
    return df

def plot_metrics_with_val(csv_path='logs/metrics.csv', output_path='logs/metrics_val_plot.png'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Train the model first.")
        return

    # Gunakan fungsi parse manual kita karena pandas read_csv akan error dengan kolom campur aduk
    df = parse_mixed_csv(csv_path)
    if len(df) == 0:
        print("CSV is empty.")
        return

    # Daftar pasangan metrik (Train Column, Val Column, Title)
    metrics_pairs = [
        ('AvgLoss', 'ValLoss', 'Total Loss'),
        ('AvgBCELoss', 'ValBCELoss', 'BCE Loss'),
        ('Precision', 'ValPrecision', 'Precision'),
        ('Recall', 'ValRecall', 'Recall'),
        ('F1_Score', 'ValF1_Score', 'F1 Score'),
        ('IoU', 'ValIoU', 'IoU'),
        ('Dice', 'ValDice', 'Dice')
    ]

    # Bikin figure grid 4 baris x 2 kolom (total 8 subplot)
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i, (train_col, val_col, title) in enumerate(metrics_pairs):
        ax = axes[i]
        
        # Plot Metrik Train
        ax.plot(df['Epoch'], df[train_col], marker='o', label=f'Train {title}', color='blue')
        
        # Plot Metrik Validasi jika datanya ada
        if not df[val_col].isna().all():
            valid_val_data = df.dropna(subset=[val_col])
            ax.plot(valid_val_data['Epoch'], valid_val_data[val_col], marker='s', label=f'Val {title}', color='red')
            
        ax.set_title(f'Train vs Val: {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        
        if len(df['Epoch']) <= 20:
            ax.set_xticks(df['Epoch'])
            
        # Batasi sumbu Y dari 0 sampai 1 untuk metrik performa (selain Loss)
        if 'Loss' not in title:
            ax.set_ylim(0, 1.05)
            
        ax.legend()
        ax.grid(True)
    
    # Hapus subplot ke-8 karena kita cuma punya 7 pasang metrik
    fig.delaxes(axes[7])
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot berhasil disimpan di: {output_path}")

if __name__ == '__main__':
    plot_metrics_with_val()
