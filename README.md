# 🌊 River Water Quality ML Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Predicting ammonium (NH4) pollution levels in river water using real monitoring data from 8 consecutive stations (1996–2019).

---

## 📌 Key Research Finding

> **Station 32 NH4 levels exceeded the safe limit (0.5 mg/dm³) by up to 44x after 2010, driven by a local pollution source independent of upstream stations — strongly suggesting industrial or urban discharge.**

![Research Findings](research_findings.png)

---

## 📖 About the Project

This project analyzes real river water quality monitoring data from Ukraine's state water monitoring system. The goal is to predict ammonium (NH4) concentration at a target station using readings from upstream stations.

**Dataset:** [Datasets for river water quality prediction](https://www.kaggle.com/vbmokin/datasets-for-river-water-quality-prediction) — Kaggle  
**Author:** [Abu Sufian](https://github.com/abusufian-dev)  
**Institution:** Tongi Govt. College, Gazipur, Bangladesh

---

## 🔍 Research Questions

- What are the pollution trends at each monitoring station over 25 years?
- Can upstream station readings predict downstream pollution levels?
- Which station has the most critical pollution crisis?

---

## 📊 Results

| Model | R² Score | RMSE |
|-------|----------|------|
| Random Forest (baseline) | 0.613 | 0.979 mg/dm³ |
| Gradient Boosting | 0.654 | 0.926 mg/dm³ |
| Random Forest (Station 32) | **0.613** | **5.83 mg/dm³** |

**Key correlation findings:**
- `Year` → **0.687** (strongest predictor — pollution grew over time)
- Upstream stations → negative/near-zero correlation with Station 32
- This confirms Station 32 has an **independent local pollution source**

---

## 🛠️ Tools & Technologies

- **Python 3.10**
- **Pandas** — data cleaning & manipulation
- **Scikit-learn** — ML models (Random Forest, Gradient Boosting)
- **Matplotlib & Seaborn** — data visualization
- **Google Colab** — development environment

---

## 📁 Project Structure

```
river-water-quality-ml/
│
├── water_quality_analysis.py   # Full analysis script
├── PB_1996_2019_NH4.csv        # Dataset (NH4 monitoring data)
├── research_findings.png       # Key research visualization
└── README.md                   # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/abusufian-dev/river-water-quality-ml.git
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the analysis:
```bash
python water_quality_analysis.py
```

---

## 🌍 Why This Matters

Ammonium pollution in rivers is a serious environmental issue. Early detection and prediction of pollution levels can help authorities take action before levels become dangerous. Bangladesh, surrounded by rivers, faces similar water quality challenges — making this research personally relevant.

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

*Made with 💙 by Abu Sufian | Class 11, Tongi Govt. College, Bangladesh*
