# 📝 Exercices et Travaux par Phase

Ce dossier contient tes exercices, projets et fichiers de travail organisés par phase.

## 📋 Structure

```
envs/
├── phase_0_foundations/     → Exercices Python basics, Jupyter
├── phase_1_math/            → Exercices maths, calculs NumPy
├── phase_2_datascience/     → Exercices Pandas, visualisation
├── phase_3_ml_classic/      → Projets ML, notebooks scikit-learn
├── phase_4_deeplearning/    → Projets PyTorch, réseaux de neurones
├── phase_5_nlp_transformers/→ Projets NLP, fine-tuning
├── phase_6_llm_dev/         → Projets LLM, RAG, agents
└── phase_7_advanced/        → Projets avancés, finance quant
```

## 🎯 Comment utiliser

1. **Lis le cours** dans `cours/Phase_X_xxx/`
2. **Fais les exercices** ici dans `envs/phase_X_xxx/`
3. **Sauvegarde ton environnement** à chaque étape importante :

```bash
conda activate llm
conda env export --no-builds > envs/phase_X_xxx/environment.yml
```

## 📁 Organisation suggérée par dossier

```
phase_X_xxx/
├── environment.yml          → Config conda (optionnel)
├── exercice_01.ipynb        → Tes exercices
├── exercice_02.ipynb
├── projet_xxx.ipynb         → Tes projets
└── notes.md                 → Tes notes personnelles
```

## 💡 Conseils

- Nomme tes fichiers clairement : `exercice_01_puissances.ipynb`
- Ajoute des commentaires dans ton code
- Commit régulièrement tes exercices sur Git
- Compare tes solutions avec les cours
