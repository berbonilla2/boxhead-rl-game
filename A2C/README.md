# A2C Training System for Boxhead Game

This folder contains an optimized A2C (Advantage Actor-Critic) training system with enhanced feature engineering and hyperparameter tuning.

## 📁 Folder Structure

```
A2C/
├── Models/                          # Trained model versions
│   ├── boxhead_A2C_v1.zip          # Baseline model
│   ├── boxhead_A2C_v2.zip          # Optimized model (generated after training)
│   └── vecnormalize_v*.pkl         # Normalization statistics
├── logs/                            # Training logs and monitoring data
├── results/                         # Training plots and metrics
├── checkpoints/                     # Model checkpoints during training
├── enhanced_boxhead_env.py         # Enhanced environment with 25 features
├── train_a2c_optimized.py          # Optimized training script
├── model_log.txt                   # Detailed log of all model versions
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install gymnasium stable-baselines3 torch numpy matplotlib pandas
```

### 2. Run Training

```bash
cd A2C
python train_a2c_optimized.py
```

The script will:
- Train A2C with optimized hyperparameters
- Save checkpoints every 10,000 steps
- Evaluate the model every 5,000 steps
- Generate real-time training plots
- Save the final model as `boxhead_A2C_v2.zip`

### 3. Monitor Training

During training, you'll see:
- Real-time plots showing rewards, losses, learning rate, kills, and entropy
- Console output with episode statistics
- Best model updates when performance improves

## 📊 Model Versions

### Version 1 (Baseline)
- **Features**: 9 basic features
- **Architecture**: Default SB3 (64, 64)
- **Learning Rate**: 7e-4 (cosine decay)
- **N-steps**: 8
- **Episodes**: 50
- **Status**: Baseline model

### Version 2 (Optimized)
- **Features**: 25 enhanced features (multi-enemy tracking, spatial awareness, action history)
- **Architecture**: Custom (128→256→256→256) + [256, 128] policy/value heads
- **Learning Rate**: 3e-4 (with warmup + cosine decay)
- **N-steps**: 16
- **Entropy**: 0.02
- **Gamma**: 0.995
- **Episodes**: 100
- **Normalization**: VecNormalize (observations + rewards)
- **Status**: Enhanced with feature engineering

## 🎯 Key Improvements in v2

1. **Enhanced Observation Space (25 features)**:
   - Player state with velocity
   - Multi-enemy tracking (1st and 2nd nearest)
   - Spatial awareness (wall distances)
   - Action history for temporal patterns
   - Enemy counts

2. **Custom Network Architecture**:
   - Deeper feature extractor
   - Layer normalization
   - Dropout for regularization
   - Separate policy and value networks

3. **Optimized Hyperparameters**:
   - Learning rate with warmup
   - Increased n-steps for better credit assignment
   - Higher entropy for exploration
   - Observation/reward normalization

4. **Better Reward Shaping**:
   - Optimal distance rewards
   - Movement incentives
   - Wall avoidance
   - Survival bonuses

5. **Training Enhancements**:
   - Checkpoint saving
   - Best model tracking
   - Comprehensive metrics logging
   - Real-time visualization

## 📈 Expected Performance

**v1 (Baseline)**:
- Mean Eval Reward: ~-20 to 0
- Mean Episode Length: 400-600 steps
- Training Time: ~5-10 minutes

**v2 (Optimized)**:
- Expected Mean Eval Reward: 10-30+ (significantly improved)
- Expected Episode Length: 700-1000 steps (longer survival)
- Training Time: ~20-40 minutes
- More consistent performance with lower variance

## 🔧 Customization

### Modify Hyperparameters

Edit `train_a2c_optimized.py`:

```python
# Line ~255: Change learning rate
learning_rate=warmup_cosine_lr(3e-4),  # Try 1e-4 or 5e-4

# Line ~256: Change n-steps
n_steps=16,  # Try 32 or 8

# Line ~260: Change entropy
ent_coef=0.02,  # Try 0.01 or 0.05

# Line ~257: Change gamma
gamma=0.995,  # Try 0.99 or 0.999
```

### Change Training Duration

```python
# Line ~33-34
EPISODES = 100  # Increase to 200 for longer training
STEPS_PER_EPISODE = 1000  # Increase to 1500 for harder episodes
```

### Update Model Version

Before training a new model:

```python
# Line ~32
MODEL_VERSION = "v3"  # Increment version number
```

## 📝 Model Versioning

Each time you train a new model:
1. Update `MODEL_VERSION` in `train_a2c_optimized.py`
2. The model will be saved as `boxhead_A2C_v{VERSION}.zip`
3. Training details are automatically appended to `model_log.txt`

## 🎮 Using Trained Models

To load and use a trained model:

```python
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from enhanced_boxhead_env import EnhancedBoxheadEnv

# Load environment with normalization
env = DummyVecEnv([lambda: EnhancedBoxheadEnv()])
env = VecNormalize.load("A2C/Models/vecnormalize_v2.pkl", env)

# Load model
model = A2C.load("A2C/Models/boxhead_A2C_v2.zip")

# Run inference
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## 📊 Analyzing Results

After training, check:

1. **Training Plots**: `results/training_v2_*.png`
   - Episode rewards over time
   - Episode lengths
   - Training losses
   - Learning rate schedule
   - Kills per episode
   - Policy entropy

2. **Metrics CSV**: `results/metrics_v2_*.csv`
   - Detailed episode-by-episode data
   - Can be analyzed in Excel/Python

3. **Model Log**: `model_log.txt`
   - Complete training configuration
   - Evaluation results
   - Observations and next steps

## 🐛 Troubleshooting

**Training is unstable:**
- Reduce learning rate (try 1e-4)
- Reduce entropy coefficient (try 0.01)
- Increase n-steps (try 32)

**Model not improving:**
- Check reward shaping in `enhanced_boxhead_env.py`
- Increase training duration (more episodes)
- Try different network architecture

**Out of memory:**
- Reduce batch size
- Use smaller network architecture
- Disable normalization plots during training

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [A2C Paper](https://arxiv.org/abs/1602.01783)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 🤝 Contributing

To add a new model version:
1. Create a new branch with feature name
2. Update `MODEL_VERSION` in the training script
3. Document changes in code comments
4. Run training and evaluation
5. Update this README with results

---

**Last Updated**: October 18, 2025
**Current Version**: v2
**Status**: Ready for training

