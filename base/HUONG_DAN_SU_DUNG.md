# 🐍 HƯỚNG DẪN SỬ DỤNG SNAKE AI SYSTEM

## 📋 **MỤC LỤC**
1. [Cài Đặt](#cài-đặt)
2. [Cấu Trúc Hệ Thống](#cấu-trúc-hệ-thống)
3. [Lệnh Training](#lệnh-training)
4. [Lệnh Chơi Game](#lệnh-chơi-game)
5. [Lệnh Đánh Giá](#lệnh-đánh-giá)
6. [Quản Lý Model](#quản-lý-model)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 **CÀI ĐẶT**

### **1. Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

### **2. Kiểm tra cài đặt:**
```bash
python -c "import pygame, torch, numpy; print('✅ All dependencies installed!')"
```

---

## 📁 **CẤU TRÚC HỆ THỐNG**

```
base/
├── 🎮 GAME FILES
│   ├── game_level1.py          # Level 1 (6 enemies, dễ)
│   ├── game_level2.py          # Level 2 (12 enemies, khó)
│   └── game_level3.py          # Level 3 (18 enemies, rất khó)
│
├── 🧠 AI FILES
│   ├── advanced_trainer.py     # AI training system
│   ├── model.py               # Neural network architecture
│   ├── trainer.py             # DQN trainer
│   └── trainer2.py            # Double DQN trainer
│
├── 🎯 EXECUTION FILES
│   ├── train_single.py        # Train một level
│   ├── play_game.py          # Chơi game với AI
│   └── play_trained_agent.py # Test AI đã train
│
└── 📊 DATA FOLDERS
    ├── saved_models/         # Models đã train
    ├── saved_experiences/    # Experience replay data
    ├── training_results/     # Kết quả training
    └── plots/               # Biểu đồ training
```

---

## 🚀 **LỆNH TRAINING**

### **1. Train Cơ Bản**
```bash
python train_single.py <Level> <Algorithm> <Episodes> <EvalInterval>
```

#### **Tham số:**
- **Level**: `Level1`, `Level2`, `Level3`
- **Algorithm**: `DQN`, `Double_DQN`
- **Episodes**: Số episodes train (VD: 2000, 5000)
- **EvalInterval**: Đánh giá mỗi X episodes (VD: 200, 500)

#### **Ví dụ:**
```bash
# Train Level1 với DQN, 2000 episodes, eval mỗi 200 episodes
python train_single.py Level1 DQN 2000 200

# Train Level2 với Double DQN, 5000 episodes, eval mỗi 500 episodes
python train_single.py Level2 Double_DQN 5000 500

# Train Level3 (khó nhất), 10000 episodes
python train_single.py Level3 DQN 10000 1000
```

### **2. Train Nâng Cao**

#### **Continue Training (tiếp tục từ model cũ):**
```bash
# Hệ thống tự động load model cũ nếu có
python train_single.py Level2 DQN 3000 300
```

#### **Train với Best Model Saving:**
```bash
# Hệ thống tự động save best model
python train_single.py Level1 DQN 2000 200
# → Tạo: DQN_Level1.pth (latest) và DQN_Level1_BEST.pth (best)
```

---

## 🎮 **LỆNH CHƠI GAME**

### **1. Chơi với AI đã train**
```bash
python play_game.py <Level> <Algorithm> <NumGames> <ShowDisplay> [ModelType]
```

#### **Tham số:**
- **Level**: `Level1`, `Level2`, `Level3`
- **Algorithm**: `DQN`, `Double_DQN`
- **NumGames**: Số games chơi (VD: 1, 5, 10)
- **ShowDisplay**: `True` (hiển thị), `False` (không hiển thị)
- **ModelType**: `latest` (mặc định), `best` (model tốt nhất)

#### **Ví dụ:**
```bash
# Chơi 1 game Level1 với hiển thị (model latest)
python play_game.py Level1 DQN 1 True

# Chơi 5 games Level2 không hiển thị (test nhanh)
python play_game.py Level2 DQN 5 False

# Chơi với best model
python play_game.py Level1 DQN 3 True best

# Test performance Level3
python play_game.py Level3 Double_DQN 10 False latest
```

### **2. Chơi Interactive (xem AI chơi từng bước)**
```bash
# Chơi 1 game với hiển thị để xem AI strategy
python play_game.py Level2 DQN 1 True best
```

---

## 📊 **LỆNH ĐÁNH GIÁ**

### **1. Đánh giá Performance**
```bash
# Tạo script đánh giá nhanh
python -c "
from advanced_trainer import GameEvaluator, AdvancedAgent
from game_level2 import Level2AI

agent = AdvancedAgent('DQN')
agent.load_model('saved_models/DQN_Level2.pth')
evaluator = GameEvaluator()
result = evaluator.evaluate_agent(agent, Level2AI, 'Level2', 20)
print(f'Win Rate: {result["win_rate"]:.1%}')
print(f'Avg Score: {result["avg_score"]:.1f}')
"
```

### **2. So sánh Models**
```bash
# So sánh latest vs best model
python play_game.py Level2 DQN 10 False latest
python play_game.py Level2 DQN 10 False best
```

---

## 🗂️ **QUẢN LÝ MODEL**

### **1. Kiểm tra Models có sẵn**
```bash
# Windows
dir saved_models\

# Linux/Mac
ls saved_models/
```

### **2. Cấu trúc tên file:**
```
saved_models/
├── DQN_Level1.pth           # Latest model Level1
├── DQN_Level1_BEST.pth      # Best model Level1
├── DQN_Level2.pth           # Latest model Level2
├── DQN_Level2_BEST.pth      # Best model Level2
├── Double_DQN_Level1.pth    # Double DQN models
└── ...
```

### **3. Backup Models**
```bash
# Backup models quan trọng
copy saved_models\DQN_Level2_BEST.pth saved_models\DQN_Level2_BACKUP.pth
```

### **4. Xóa Models cũ**
```bash
# Xóa models không cần thiết (cẩn thận!)
del saved_models\DQN_Level1.pth
```

---

## 📈 **XEM KẾT QUẢ TRAINING**

### **1. Xem Plots**
```bash
# Mở folder plots để xem biểu đồ
explorer plots\        # Windows
open plots/           # Mac
nautilus plots/       # Linux
```

### **2. Xem Training Results**
```bash
# Xem JSON results
type training_results\DQN_Level2_results.json    # Windows
cat training_results/DQN_Level2_results.json     # Linux/Mac
```

---

## 🎯 **WORKFLOW KHUYẾN NGHỊ**

### **1. Workflow cho người mới:**
```bash
# Bước 1: Train Level1 (dễ nhất)
python train_single.py Level1 DQN 2000 200

# Bước 2: Test model
python play_game.py Level1 DQN 5 True best

# Bước 3: Nếu win rate > 20%, chuyển Level2
python train_single.py Level2 DQN 3000 300

# Bước 4: Test Level2
python play_game.py Level2 DQN 10 False best
```

### **2. Workflow cho người có kinh nghiệm:**
```bash
# Train multiple levels song song
python train_single.py Level1 DQN 5000 500 &
python train_single.py Level2 Double_DQN 5000 500 &
python train_single.py Level3 DQN 10000 1000 &
```

### **3. Workflow tối ưu performance:**
```bash
# 1. Train với episodes cao
python train_single.py Level2 DQN 10000 500

# 2. Test với sample lớn
python play_game.py Level2 DQN 50 False best

# 3. So sánh algorithms
python train_single.py Level2 Double_DQN 10000 500
```

---

## 🔧 **TROUBLESHOOTING**

### **❌ Lỗi thường gặp:**

#### **1. "Model not found"**
```bash
# Kiểm tra file có tồn tại không
ls saved_models/DQN_Level1.pth

# Nếu không có, train lại
python train_single.py Level1 DQN 1000 100
```

#### **2. "CUDA out of memory"**
```bash
# Giảm batch size trong advanced_trainer.py
# Hoặc dùng CPU
export CUDA_VISIBLE_DEVICES=""
```

#### **3. "Pygame error"**
```bash
# Cài lại pygame
pip uninstall pygame
pip install pygame
```

#### **4. "Size mismatch error"**
```bash
# Model cũ không tương thích, xóa và train lại
del saved_models\DQN_Level1.pth
python train_single.py Level1 DQN 2000 200
```

### **⚠️ Lưu ý quan trọng:**

#### **1. Training Time:**
- **Level1**: ~30-60 phút (2000 episodes)
- **Level2**: ~1-2 giờ (5000 episodes)  
- **Level3**: ~2-4 giờ (10000 episodes)

#### **2. Memory Usage:**
- **RAM**: ~2-4GB khi training
- **Disk**: ~100MB cho models và data

#### **3. Performance Expected:**
- **Level1**: Win rate 30-60% sau training
- **Level2**: Win rate 10-30% sau training
- **Level3**: Win rate 5-15% sau training

---

## 📚 **EXAMPLES NÂNG CAO**

### **1. Batch Training Script:**
```bash
# Tạo file train_all.bat (Windows)
echo python train_single.py Level1 DQN 3000 300 > train_all.bat
echo python train_single.py Level1 Double_DQN 3000 300 >> train_all.bat
echo python train_single.py Level2 DQN 5000 500 >> train_all.bat
echo python train_single.py Level2 Double_DQN 5000 500 >> train_all.bat

# Chạy
train_all.bat
```

### **2. Performance Testing Script:**
```bash
# Tạo file test_all.py
cat > test_all.py << 'EOF'
import subprocess
import sys

levels = ['Level1', 'Level2', 'Level3']
algorithms = ['DQN', 'Double_DQN']

for level in levels:
    for algo in algorithms:
        print(f"\n🧪 Testing {algo} on {level}")
        try:
            result = subprocess.run([
                sys.executable, 'play_game.py', 
                level, algo, '10', 'False', 'best'
            ], capture_output=True, text=True)
            print(result.stdout)
        except Exception as e:
            print(f"❌ Error: {e}")
EOF

python test_all.py
```

### **3. Model Comparison:**
```bash
# So sánh performance của tất cả models
python -c "
import os
models = [f for f in os.listdir('saved_models') if f.endswith('.pth')]
print('📁 Available Models:')
for model in sorted(models):
    size = os.path.getsize(f'saved_models/{model}') / 1024 / 1024
    print(f'  {model:<25} ({size:.1f} MB)')
"
```

---

## 🎉 **KẾT LUẬN**

Hệ thống Snake AI này cung cấp:
- ✅ **3 levels** với độ khó tăng dần
- ✅ **2 algorithms** (DQN, Double DQN)
- ✅ **Best model saving** tự động
- ✅ **Detailed evaluation** và visualization
- ✅ **Easy-to-use commands** cho mọi use case

**Happy Training!** 🚀🐍🤖 