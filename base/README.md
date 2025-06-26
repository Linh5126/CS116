<<<<<<< HEAD
# 🐍 Snake AI System - Hệ Thống AI Chơi Snake

Hệ thống Reinforcement Learning để train AI agents chơi Snake game sử dụng PyTorch và Pygame với DQN/Double DQN algorithms.

## 🎯 **FEATURES**
- ✅ **3 Levels** với độ khó tăng dần (6, 12, 18 enemies)
- ✅ **2 Algorithms**: DQN và Double DQN
- ✅ **Best Model Saving** tự động
- ✅ **Real-time visualization** và performance tracking
- ✅ **Detailed evaluation** với charts và metrics

## 📚 **DOCUMENTATION**

### **📖 Hướng dẫn chi tiết:**
- **[HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md)** - Hướng dẫn đầy đủ tất cả lệnh
- **[QUICK_COMMANDS.md](QUICK_COMMANDS.md)** - Lệnh nhanh và workflow
- **[COMMANDS_BY_SCENARIO.md](COMMANDS_BY_SCENARIO.md)** - Lệnh theo tình huống cụ thể

## ⚡ **QUICK START**

### **1. Cài đặt:**
```bash
pip install -r requirements.txt
```

### **2. Train AI đầu tiên:**
```bash
# Train Level1 (dễ nhất)
python train_single.py Level1 DQN 2000 200
```

### **3. Chơi với AI:**
```bash
# Xem AI chơi
python play_game.py Level1 DQN 1 True best
```

## 🎮 **LEVELS**
- **Level1**: 6 enemies (dễ) - Win rate expected: 30-60%
- **Level2**: 12 enemies (khó) - Win rate expected: 10-30%  
- **Level3**: 18 enemies (rất khó) - Win rate expected: 5-15%

## 🧠 **ALGORITHMS**
- **DQN**: Deep Q-Network cơ bản
- **Double_DQN**: Double DQN với target network

## 📊 **EXPECTED PERFORMANCE**
- **Training time**: 30 phút - 4 giờ tùy level
- **Memory usage**: 2-4GB RAM
- **Disk space**: ~100MB cho models và data

## 🚀 **ADVANCED USAGE**
Xem các file documentation để biết thêm về:
- Continue training từ model cũ
- So sánh algorithms
- Debugging và troubleshooting
- Production deployment

**Happy Training!** 🎉🤖

---

## 📺 **ORIGINAL TUTORIAL**
Based on Python Reinforcement Learning Tutorial series: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)
=======
# Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame

In this Python Reinforcement Learning Tutorial series we teach an AI to play Snake! We build everything from scratch using Pygame and PyTorch. The tutorial consists of 4 parts:

You can find all tutorials on my channel: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)

- Part 1: I'll show you the project and teach you some basics about Reinforcement Learning and Deep Q Learning.
- Part 2: Learn how to setup the environment and implement the Snake game.
- Part 3: Implement the agent that controls the game.
- Part 4: Implement the neural network to predict the moves and train it.
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
