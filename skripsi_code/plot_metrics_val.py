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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: Losses ---
    axes[0].plot(df['Epoch'], df['AvgLoss'], marker='o', label='Train Total Loss', color='blue', linestyle='-')
    axes[0].plot(df['Epoch'], df['AvgBCELoss'], marker='x', label='Train BCE Loss', color='lightblue', linestyle='--')
    
    # Gambar plot Validasi jika datanya ada (mengabaikan nilai None/NaN)
    if not df['ValLoss'].isna().all():
        # DropNa hanya untuk series yang digambar agar matplotlib tidak memotong garisnya
        valid_val_data = df.dropna(subset=['ValLoss'])
        axes[0].plot(valid_val_data['Epoch'], valid_val_data['ValLoss'], marker='s', label='Val Total Loss', color='red', linestyle='-')
        axes[0].plot(valid_val_data['Epoch'], valid_val_data['ValBCELoss'], marker='^', label='Val BCE Loss', color='salmon', linestyle='--')
        
    axes[0].set_title('Training & Validation Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    if len(df['Epoch']) <= 20:
        axes[0].set_xticks(df['Epoch'])
    axes[0].legend()
    axes[0].grid(True)
    
    # --- Plot 2: Performance Metrics ---
    # Metrik Train
    axes[1].plot(df['Epoch'], df['F1_Score'], marker='o', label='Train F1 Score', color='blue')
    axes[1].plot(df['Epoch'], df['IoU'], marker='x', label='Train IoU', color='lightblue')
    
    # Metrik Val
    if not df['ValF1_Score'].isna().all():
        valid_val_f1 = df.dropna(subset=['ValF1_Score'])
        axes[1].plot(valid_val_f1['Epoch'], valid_val_f1['ValF1_Score'], marker='s', label='Val F1 Score', color='red')
        axes[1].plot(valid_val_f1['Epoch'], valid_val_f1['ValIoU'], marker='^', label='Val IoU', color='salmon')
        
    axes[1].set_title('Training & Validation Metrics over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score (0 to 1)')
    if len(df['Epoch']) <= 20:
        axes[1].set_xticks(df['Epoch'])
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Plot berhasil disimpan di: {output_path}")

if __name__ == '__main__':
    plot_metrics_with_val()
