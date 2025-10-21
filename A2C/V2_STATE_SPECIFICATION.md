# A2C V2 - Comprehensive State Representation

## 📋 Overview

The A2C V2 model features a **comprehensive 60-feature state representation** that includes all requested state components for advanced tactical gameplay.

---

## 🎯 State Components (60 Features Total)

### 1. **Position** (Features 0-2)
Agent's current location and orientation on the map:
- `[0]` **Player X**: Normalized X coordinate [-1, 1]
- `[1]` **Player Y**: Normalized Y coordinate [-1, 1]
- `[2]` **Facing Direction**: Current direction the agent is facing

**Purpose**: Core spatial awareness for navigation and movement decisions

---

### 2. **Enemy Information** (Features 3-32)
Positions and health of the **5 nearest enemies** (30 features total):

**For each enemy (6 features × 5 enemies):**
- **Delta X**: Relative X position to player
- **Delta Y**: Relative Y position to player
- **Distance**: Euclidean distance to player (normalized)
- **Enemy Type**: Zombie (+1.0) or Demon (-1.0)
- **Health**: Current health ratio (normalized)
- **Speed**: Movement speed of the enemy

**Enemies Tracked**:
- **Zombies**: Health 30, Speed 0.7, easier to kill
- **Demons**: Health 50, Speed 0.9, more dangerous

**Purpose**: Tactical awareness of immediate threats for combat and evasion

---

### 3. **Agent Status** (Features 33-36)
Current health level and combat performance:
- `[33]` **Current Health**: Player health (0-100, normalized)
- `[34]` **Damage Taken**: Cumulative damage received this episode
- `[35]` **Kill Count**: Number of enemies killed (soft normalized)
- `[36]` **Shot Accuracy**: Hit rate (hits/shots fired)

**Purpose**: Self-monitoring for survival decisions and combat effectiveness

---

### 4. **Resources** (Features 37-41)
Ammo count and currently equipped weapon:
- `[37]` **Ammo Count**: Current ammunition (normalized, max 100)
- `[38]` **Has Pistol**: Binary flag (+1.0 if equipped)
- `[39]` **Has Shotgun**: Binary flag (+1.0 if equipped)
- `[40]` **Has Rifle**: Binary flag (+1.0 if equipped)
- `[41]` **Weapons Unlocked**: Ratio of unlocked weapons

**Weapon Statistics**:
| Weapon | Damage | Ammo Cost | Range |
|--------|--------|-----------|-------|
| Pistol | 25 | 1 | 150 |
| Shotgun | 50 | 2 | 100 |
| Rifle | 35 | 1 | 200 |

**Purpose**: Resource management and weapon selection strategy

---

### 5. **Items** (Features 42-47)
Locations of nearby pickups:

**Ammo Pickups (Features 42-44)**:
- `[42]` **Nearest Ammo Delta X**: Relative X position
- `[43]` **Nearest Ammo Delta Y**: Relative Y position
- `[44]` **Distance to Ammo**: Normalized distance

**Weapon Pickups (Features 45-47)**:
- `[45]` **Nearest Weapon Delta X**: Relative X position
- `[46]` **Nearest Weapon Delta Y**: Relative Y position
- `[47]` **Distance to Weapon**: Normalized distance

**Item Types**:
- **Ammo Pickups**: +20 ammunition
- **Weapon Pickups**: Unlock shotgun or rifle

**Purpose**: Strategic resource acquisition planning

---

### 6. **Map Layout** (Features 48-55)
Static features such as walls, chokepoints, and open areas:

**Wall Distances (Features 48-51)**:
- `[48]` **Distance to Left Wall**: How far from left boundary
- `[49]` **Distance to Right Wall**: How far from right boundary
- `[50]` **Distance to Top Wall**: How far from top boundary
- `[51]` **Distance to Bottom Wall**: How far from bottom boundary

**Strategic Features (Features 52-55)**:
- `[52]` **Nearest Chokepoint Distance**: Distance to narrow passage
- `[53]` **Nearest Open Area Distance**: Distance to safe zone
- `[54]` **In Chokepoint**: Binary flag (+1.0 if inside)
- `[55]` **In Open Area**: Binary flag (+1.0 if inside)

**Map Features**:
- **Walls**: Outer boundaries + 2 internal pillars (create obstacles)
- **Chokepoints**: 3 narrow passages (tactical positions)
- **Open Areas**: 4 safe zones (retreat options)

**Purpose**: Tactical positioning and terrain utilization

---

### 7. **Action History** (Features 56-59)
Last 4 actions taken (temporal awareness):
- `[56]` **Action t-3**: Action taken 3 steps ago
- `[57]` **Action t-2**: Action taken 2 steps ago
- `[58]` **Action t-1**: Action taken 1 step ago
- `[59]` **Action t-0**: Most recent action

**Purpose**: Temporal patterns and action sequence learning

---

## 🎮 Action Space

6 Discrete Actions:
- **0**: Idle/No action
- **1**: Move Up
- **2**: Move Down
- **3**: Move Left
- **4**: Move Right
- **5**: Shoot (with current weapon)

---

## 🏗️ Network Architecture

### Feature Extractor
```
Input (60 features)
    ↓
Linear(60 → 256) + LayerNorm + ReLU + Dropout(0.1)
    ↓
Linear(256 → 512) + LayerNorm + ReLU + Dropout(0.1)
    ↓
Linear(512 → 512) + LayerNorm + ReLU + Dropout(0.08)
    ↓
Linear(512 → 512) + LayerNorm + ReLU + Dropout(0.05)
    ↓
Linear(512 → 512) + LayerNorm + ReLU
    ↓
Output: 512-dimensional features
```

### Policy & Value Heads
- **Policy Network**: [512 → 256 → 128 → 6 actions]
- **Value Network**: [512 → 256 → 128 → 1 value]

**Total Parameters**: ~2M+ (significantly larger than v1)

---

## ⚙️ Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-4 | Lower for complex state space |
| Warmup | 15% of training | Longer warmup for stability |
| N-steps | 24 | Better credit assignment |
| Gamma | 0.997 | Long-term planning |
| Entropy | 0.025 | More exploration needed |
| Episodes | 150 | Extended training |
| Steps/Episode | 1000 | Longer episodes |
| Total Timesteps | 150,000 | 3.75x more than v1 |

---

## 🎁 Reward Structure

### Positive Rewards
- **Survival**: +0.1 per step
- **Health maintenance**: +0.2 × (health/100)
- **Optimal positioning**: +0.4 (80-150 pixels from enemy)
- **Kill demon**: +8.0
- **Kill zombie**: +5.0
- **Hit enemy**: +1.0
- **Ammo pickup**: +3.0
- **Weapon pickup**: +5.0
- **Complete episode**: +25

### Penalties
- **Death**: -50
- **Too close to enemy**: -0.3
- **Low ammo**: -0.2
- **Hit wall**: -0.05
- **Miss shot**: -0.05
- **Shoot without ammo**: -0.1
- **Enemy collision**: -0.8

---

## 📊 Key Improvements Over Previous Versions

### From v1 (9 features) → V2 (60 features)

| Aspect | v1 | V2 | Improvement |
|--------|----|----|-------------|
| Features | 9 | **60** | **+567%** |
| Enemy Tracking | 1 nearest | **5 nearest** | **5x more** |
| Weapon System | None | **3 weapons** | **New** |
| Ammo System | None | **Full management** | **New** |
| Items | None | **Ammo + Weapons** | **New** |
| Map Features | Basic | **Walls + Chokepoints** | **New** |
| Combat Stats | None | **Accuracy tracking** | **New** |
| Network Depth | 2 layers | **5 layers** | **2.5x deeper** |
| Hidden Units | 64 | **512** | **8x larger** |
| Total Training | 40k steps | **150k steps** | **3.75x more** |

---

## 🚀 Usage

### Train the Model
```bash
# Activate virtual environment
gameEnv\Scripts\activate

# Navigate to A2C folder
cd A2C

# Run training (DO NOT RUN YET - as requested)
python A2C_V2_train.py
```

### What Training Will Do
1. Initialize 60-feature environment
2. Train for 150 episodes (150,000 timesteps)
3. Track 10+ metrics including accuracy
4. Save checkpoints every 10,000 steps
5. Evaluate every 5,000 steps
6. Generate comprehensive plots
7. Update model log automatically

### Expected Outputs
- `Models/boxhead_A2C_v2.zip` - Trained model
- `Models/vecnormalize_v2.pkl` - Normalization stats
- `results/training_v2_{timestamp}.png` - Training plots
- `results/metrics_v2_{timestamp}.csv` - Episode data
- `checkpoints/` - Training checkpoints
- `model_log.txt` - Updated with v2 results

---

## 📈 Expected Performance

### Compared to v1
- **Reward**: Expected 100-200% improvement
- **Survival**: Expected 50-80% longer episodes
- **Strategy**: Much more sophisticated behavior
- **Resource Management**: Strategic ammo/weapon usage
- **Map Usage**: Tactical positioning in chokepoints/open areas
- **Combat**: Higher accuracy and kill efficiency

### Success Indicators
✅ Uses map features strategically (avoids walls, uses chokepoints)  
✅ Manages ammo efficiently (picks up ammo before running out)  
✅ Selects appropriate weapons for situations  
✅ Maintains optimal distance from enemies  
✅ Higher shot accuracy (>50%)  
✅ Longer survival times  
✅ More kills per episode  

---

## 🔍 State Visualization

```
STATE VECTOR (60 features):

[Position: 3 features]
├─ Player X, Y coordinates
└─ Facing direction

[Enemy Info: 30 features (5 enemies × 6 features each)]
├─ Enemy 1 (nearest): DX, DY, Dist, Type, Health, Speed
├─ Enemy 2: DX, DY, Dist, Type, Health, Speed
├─ Enemy 3: DX, DY, Dist, Type, Health, Speed
├─ Enemy 4: DX, DY, Dist, Type, Health, Speed
└─ Enemy 5: DX, DY, Dist, Type, Health, Speed

[Agent Status: 4 features]
├─ Health
├─ Damage taken
├─ Kills
└─ Accuracy

[Resources: 5 features]
├─ Ammo count
├─ Pistol equipped
├─ Shotgun equipped
├─ Rifle equipped
└─ Weapons unlocked

[Items: 6 features]
├─ Nearest ammo: DX, DY, Distance
└─ Nearest weapon: DX, DY, Distance

[Map Layout: 8 features]
├─ Wall distances: Left, Right, Top, Bottom
├─ Nearest chokepoint distance
├─ Nearest open area distance
├─ In chokepoint (binary)
└─ In open area (binary)

[Action History: 4 features]
└─ Last 4 actions (temporal awareness)
```

---

## 📝 Files Created

1. **`enhanced_boxhead_env_v2.py`** (600+ lines)
   - Full environment implementation
   - 60-feature state representation
   - Weapon system, ammo management
   - Map layout with strategic features
   - Item spawning and pickups

2. **`A2C_V2_train.py`** (700+ lines)
   - Complete training script
   - Custom network for 60 features
   - Enhanced callbacks and metrics
   - Comprehensive logging
   - Auto-documentation

3. **`V2_STATE_SPECIFICATION.md`** (This file)
   - Complete state documentation
   - Architecture details
   - Usage instructions

---

## ⚠️ Important Notes

**DO NOT TRAIN YET** - As requested, the scripts are ready but training has not been initiated.

**Before Training**:
1. Review state representation (this document)
2. Verify hyperparameters in `A2C_V2_train.py`
3. Ensure sufficient disk space (~500MB for checkpoints)
4. Allocate 40-60 minutes for training

**During Training**:
- Monitor the 6 real-time plots
- Watch for increasing rewards and accuracy
- Check that entropy decreases gradually
- Ensure losses stabilize

**After Training**:
- Compare v2 results with v1 baseline
- Analyze weapon usage patterns
- Review accuracy improvements
- Check map utilization

---

## 🎯 Next Steps (After Training)

1. **Evaluate v2 performance** against v1 baseline
2. **Analyze new features**: weapon usage, ammo management, map utilization
3. **Tune hyperparameters** if needed based on results
4. **Create v3** with additional optimizations
5. **Experiment** with hierarchical policies or multi-agent setups

---

**Status**: ✅ **READY FOR TRAINING** (Scripts prepared, awaiting user command)  
**Model Version**: v2  
**State Features**: 60 (comprehensive)  
**Expected Training Time**: 40-60 minutes  

---

**Last Updated**: October 18, 2025  
**Files Ready**: ✅ Environment, ✅ Training Script, ✅ Documentation  
**Linting**: ✅ No errors  
**Testing**: Pending (will run after training approved)  

