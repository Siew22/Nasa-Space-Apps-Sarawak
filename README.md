cd ~/Nasa\ Space\ Apps

cat << 'EOF' > README.md
# Flame in Freefall: Physics-Informed Microgravity Combustion Research & STEM Platform

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-Sarawak%202026-blue.svg)](https://www.spaceappschallenge.org/)
[![Python Version](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://www.python.org/)
[![Tests Passing](https://img.shields.io/badge/tests-5%2F5%20passed-success.svg)](tests/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

> **NASA Space Apps Challenge Topic**: *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*  
> An interactive, physics-grounded scientific research and STEM education platform bridging NASA flight combustion archives with first-principles fluid dynamics for transparent human spaceflight fire safety.

---

## 🚀 Quick Start for Teammates & Collaborators (队友极速上手指南)

> 💡 **给队友的重要提示（没有模型怎么办？）**：  
> 本平台的交互前端、证据库与 2D 物理沙盒**开箱即用，不需要预先下载任何大模型权重文件**！  
> 核心分析与仿真完全由第一性原理物理引擎与 NASA 真实清洗数据驱动。如果你想自己本地训练一个深度学习模型，只需运行一条命令即可（见步骤 4）。

### 1. 环境准备 (Environment Setup)
推荐在 **Linux / WSL2 (Ubuntu) / macOS** 环境下使用 **Python 3.11**：

```bash
# 克隆仓库
git clone https://github.com/Siew22/Nasa-Space-Apps-Sarawak.git
cd Nasa-Space-Apps-Sarawak

# 创建并激活虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 安装严格锁定的依赖包 (锁定 pyarrow==16.1.0 与 numpy==1.26.4，防止版本冲突)
pip install --upgrade pip
pip install -r requirements.txt
pip install "pyarrow==16.1.0"
```

### 2. 一键启动 STEM 可视化平台 (Run Interactive Dashboard)
**无需任何预训练权重**，直接启动：
```bash
streamlit run src/app.py
```
打开浏览器访问 **`http://localhost:8501`**，即可体验全部 4 大科研与教学板块！

### 3. 运行自动化物理边界测试 (Run Physics Tests)
验证 2D 模拟引擎的 5 大物理极限守恒律：
```bash
pytest tests/ -v
```
*(预期结果：5 passed in 0.2s)*

### 4. (可选) 本地训练你的深度学习模型 (Train Multi-Task Model Locally)
如果你想在本地生成自己的 TensorFlow 神经网络模型：
```bash
# 自动在 models/ 下生成 spacecraft_fire_multitask_v1.keras
python -m src.train_spacecraft_fire_model
```

### 5. (可选) 启动 FastAPI 后端服务
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
访问 `http://localhost:8000/docs` 查看完整的 OpenAPI / Swagger 接口文档。

---

## 🌌 Project Background & Challenge Problem (赛题背景与核心痛点)

在地球常重力（$1g$）环境下，热燃烧产物因密度变小而产生强烈的阿基米德浮力上升，形成向上的锥形火焰羽流。但在太空微重力（$\mu g$）环境下：
1. **浮力对流完全消失**：热烟气不上升，自然对流供氧阻断；
2. **燃烧由扩散与通风主导**：风速极低时火源因产物滞留发生**“缺氧自熄（Oxygen Starvation Quenching）”**；微弱通风则可能引发剧烈蔓延；
3. **NASA 几十年的飞行数据极度分散**：BASS（固体板）与 FLEX（液滴）处于完全不同的物理体系。

传统黑客松项目常犯的致命错误是将不同体系的数据“一锅炖”训练黑盒大模型，导致**捷径学习（Shortcut Learning）**与数据泄漏。  
**本项目坚决拒绝虚假硬编码评分，构建了真实分轨治理、第一性原理驱动的科研分析平台。**

---

## 🔬 Core System Architecture (平台四大核心板块)

```text
Nasa-Space-Apps-Sarawak/
├── 🔬 Tab 1: NASA 实验证据库 (BASS-II vs FLEX 分轨治理与真实聚合统计)
├── 📈 Tab 2: STEM 科研变量敏感性分析 (理查森数双轴响应曲线 + 物理因果叙述)
├── 🧪 Tab 3: 2D 微重力物理时空沙盒 (高斯温度场时空演化网格 + 质心动力学)
└── 📊 Tab 4: 科学审计大屏 (25 折 GroupKFold 无泄漏交叉验证报告)
```

### 1. 物理分轨证据库 (NASA Evidence Explorer - Phase 2)
* **固体表面燃烧轨道 (`solid_fuel` / BASS-II)**：DOI: `10.60555/4qc4-de67`，探索强迫气流对固体线蔓延速度与 CO/CO₂ 释放的影响。
* **空间悬浮液滴燃烧轨道 (`liquid_droplet` / FLEX)**：DOI: `10.60555/mbq8-0451`，探索环境气压与稀释对初始直径 $d_0$、熄灭直径 $d_{ext}$ 的影响。
* **绝对无硬编码**：材料火灾危险度排行完全通过 Python `groupby` 聚合 NASA 真实实验统计生成。

### 2. STEM 变量敏感性分析与物理叙事 (STEM Analysis - Phase 4)
学生可自由选择自变量（风速 $u_{forced}$、重力 $g$、氧浓度 $O_2$）执行连续参数扫描（Parametric Sweep）：
* **双轴流体响应曲线**：动态解算**理查森数 $\log_{10}(Ri)$** 与有效供氧通量（Oxygen Flux Index）；
* **物理因果机制叙述 (Scientific Narration)**：直接向学生阐明物理因果——微重力下为何无风自熄、微风助燃、大风吹熄。

### 3. 2D 微重力物理时空沙盒 (2D Physics Sandbox - Phase 3)
通过第一性原理动力学求解器计算 2D 空间网格场：
* **严格计算量**：理想气体密度差、净浮力 $F_{net}$、约化重力 $g' = g \frac{\Delta \rho}{\rho}$、特征浮力速度 $u_b = \sqrt{g' L}$、热扩散半径 $r(t) = \sqrt{r_0^2 + 2 D t}$、质心阻尼运动 $[x_c(t), y_c(t)]$；
* **低阶教学示意量**：空间 2D 高斯温度热力图（Heatmap）与速度矢量分量；
* **直观对比**：拖动时间轴与重力滑块，直观看到微重力（球对称原地扩散）瞬间转变为地球常重力（高速向上浮力羽流拉伸）。

### 4. 科学审计与 25 折验证监视器 (Scientific Audit - Phase 1)
* 针对 80 组独立实验采用 **5 种子 × 5 折 = 25 轮严格的 GroupKFold** 外层隔离测试；
* 严格断言组间零重叠（`assert len(train_groups & val_groups) == 0`），彻底杜绝数据泄漏；
* 公开对比 **Dummy 朴素基线**、**随机森林 (Random Forest)** 与 **TensorFlow 多任务网络** 的真实均值与方差（Mean ± Std）。

---

## 🧪 Scientific Boundary & Physics Sanity Checks (物理自洽性测试)

我们在 `tests/test_simulation_physics_limits.py` 中编写了严格的物理守恒测试，确保代码不搞假动画：

| 测试用例名称 | 检验的第一性原理物理规律 | 测试状态 |
| :--- | :--- | :---: |
| `test_microgravity_limit_zero_buoyancy_motion` | $g=0$ 时垂直浮力、上升位移与垂直速度恒为 0（火焰绝不违背物理向上飘） | **PASSED** ✅ |
| `test_gravity_monotonic_buoyancy_scaling` | 浮力速度尺度 $u_b$ 随 $\sqrt{g}$ 从月球到地球严格单调递增 | **PASSED** ✅ |
| `test_plume_deflection_angle_comparison` | 相同通风下，地球羽流偏竖立、月球羽流顺风倒伏（偏转角 $\theta_{earth} > \theta_{moon}$） | **PASSED** ✅ |
| `test_temporal_continuity_and_smooth_centroid` | $t=0$ 时质心位于原点，扩散半径连续平滑扩展，无突变跳跃 | **PASSED** ✅ |
| `test_unphysical_temperature_rejected` | Pydantic 物理防线：严格拒绝逆温或负浮力非法参数 | **PASSED** ✅ |

---

## 📂 Repository Structure (项目代码目录)

```text
├── app/                            # FastAPI 后端服务与核心计算模块
│   ├── main.py                     # REST API 路由与 OpenAPI 接口
│   ├── schemas.py                  # Pydantic v2 严格参数验证与 2D 网格 Schema
│   ├── settings.py                 # 动态配置与跨域管理
│   └── services/                   # 业务逻辑服务层
│       ├── data_service.py         # Phase 2: 分轨数据检索与物理字典
│       ├── simulation_service.py   # Phase 3: 2D 物理空间网格求解器
│       └── research_engine.py      # Phase 4: 参数扫描与物理叙事引擎
├── src/                            # 交互前端与数据清洗流水线
│   ├── app.py                      # Streamlit 4 大板块交互式前端大屏
│   └── generate_scientific_validation.py # Phase 1: 5x5 GroupKFold 科学审计脚本
├── config/                         # 规范配置文件
│   ├── dataset_catalog.csv         # 数据集来源、DOI 与许可清单
│   ├── scientific_validation.json  # 交叉验证配置
│   └── simulation_model.json       # 物理常量与参数上下限约束
├── data/                           # 数据目录 (真实观测数据)
│   ├── raw/                        # 原始实验表
│   └── processed/                  # 清洗后规范表 (dashboard_records.csv)
├── reports/                        # 科学审计导出报告 (可查验 CSV/JSON)
│   ├── data_audit_report.json      # 数据血统与质量审查报告
│   ├── cv_fold_metrics.csv         # 25 折独立评测指标明细
│   └── experiment_split_manifest.csv # 逐行隔离切分清单 (防作弊凭证)
├── tests/                          # 自动化测试套件
│   └── test_simulation_physics_limits.py # 5 大物理极限守恒测试
├── pytest.ini                      # 测试路径配置
└── requirements.txt                # 严格锁定的依赖清单
```

---

## 📚 Data Provenance & Citations (NASA 官方数据来源)

本项目所有真实数据完全遵循 NASA 开放科学协议（Open Science）：

1. **NASA BASS-II (Burning and Suppression of Solids II)**  
   * NASA Physical Sciences Informatics: [PSI-25](https://psi.nasa.gov/physci/repo/data/investigations/PSI-25)  
   * DOI: `10.60555/4qc4-de67` | License: **CC0-1.0**  
2. **NASA FLEX (Flame Extinguishment Experiment)**  
   * NASA Physical Sciences Informatics: [PSI-69](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69)  
   * DOI: `10.60555/mbq8-0451` | License: **CC0-1.0**  
3. **NASA Saffire Spacecraft Fire Safety Demonstration**  
   * NASA Technical Reports: NASA/TM-20205008182 / NASA/TM-20240002981  
4. **MaCFP Condensed Phase Material Database**  
   * International Association for Fire Safety Science (IAFSS): [MaCFP GitHub](https://github.com/MaCFP/matl-db)

---

## 👥 Contributors & Team

* **Team**: Sarawak Space Apps Team 2026
* **Repository**: [https://github.com/Siew22/Nasa-Space-Apps-Sarawak](https://github.com/Siew22/Nasa-Space-Apps-Sarawak)
EOF

git add README.md
git commit -m "docs: add complete scientific documentation and teammate guide"
git push origin main
echo "🎉 全部操作执行完毕！README.md 已成功生成并推送到 GitHub！"
