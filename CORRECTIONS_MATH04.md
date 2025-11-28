# Rapport de Corrections - Maths 04 (Algèbre Linéaire Avancée)

## 📋 Résumé des Corrections

Deux problèmes majeurs ont été identifiés et corrigés pour assurer la cohérence pédagogique des exercices.

---

## 1️⃣ Problème : Exercice 3.5 Utilise SVD Trop Tôt

### Situation Initiale
- **Exercice 3.5** (Section 3 : Rang) utilisait `np.linalg.svd()` pour l'approximation de rang faible
- **Problème** : SVD n'est enseigné qu'en **Section 5**
- **Impact** : Forçait les étudiants à utiliser un concept non encore couvert

### ✅ Solution Appliquée

**Étape 1 : Remplacement dans Section 3**
- Ancien Ex 3.5: "Approximation de Rang Faible (avec SVD)" → Supprimé
- Nouvel Ex 3.5: "Rang d'un Produit Matriciel"
  - Vérifie la propriété: `rang(AB) ≤ min(rang(A), rang(B))`
  - Utilise uniquement `np.linalg.matrix_rank()`
  - Concept: Inégalité du rang (niveau Section 3)

**Étape 2 : Déplacement vers Section 5**
- Créé nouveau **Exercice 5.6**: "Approximation de Rang Faible"
- Position: Après SVD (Exercices 5.1-5.5)
- Contenu: Identique à l'ancien 3.5 mais contextuel
- Utilise: `np.linalg.svd()` comme méthode enseignée

### Fichiers Modifiés
```
- exercices_04_algebre_avancee.ipynb
  ✓ Section 3, Ex 3.5 → Remplacé par rang de produit
  ✓ Section 5 → Nouvel Ex 5.6 inséré après SVD

- solutions_04_algebre_avancee.ipynb
  ✓ Solution 5.2 → Complétée avec Solution 5.6
  ✓ Incluent code et explications détaillées
```

---

## 2️⃣ Problème : Exercice 5.4 Manque d'Explication Conceptuelle

### Situation Initiale
- **Exercice 5.4** (Pseudo-Inverse) mentionnait "Moore-Penrose" sans introduction
- **Problème** : Concept avancé jamais présenté dans le cours
- **Impact** : Étudiants confus sur ce qu'est une pseudo-inverse

### ✅ Solution Appliquée

**Énoncé Enrichi**
```markdown
**Concept avancé** : La pseudo-inverse de Moore-Penrose A⁺ généralise l'inverse pour:
- Matrices non-carrées (m ≠ n)
- Matrices singulières (det(A) = 0)

Elle se calcule via SVD: A⁺ = V·Σ⁺·Uᵀ où Σ⁺ inverse les valeurs singulières
```

**Solution Enrichie**
- Code complet d'implémentation manuelle
- Comparaison avec `np.linalg.pinv()`
- Démonstration des propriétés mathématiques:
  - A·A⁺·A = A
  - A⁺·A·A⁺ = A⁺

### Fichiers Modifiés
```
- exercices_04_algebre_avancee.ipynb
  ✓ Énoncé Ex 5.4 → Ajout note explicative

- solutions_04_algebre_avancee.ipynb
  ✓ Section 5.4 → Code solution complet (22 lignes)
  ✓ Inclut verbalisation du concept et propriétés
```

---

## 📊 Impact Pédagogique

### Avant Corrections
❌ Progression non-linéaire (SVD utilisé avant d'être enseigné)
❌ Concept orphelin (pseudo-inverse sans explication)
❌ Confusion possible pour les étudiants

### Après Corrections
✅ Progression strictement séquentielle
✅ Concepts expliqués avant utilisation
✅ Déploiement logique des algorithmes

---

## 🔍 Détails Techniques

### Ex 3.5 → New (Rang du Produit)
```python
# Utilise uniquement :
np.linalg.matrix_rank(A)  # ✓ Enseigné Section 3
# PAS np.linalg.svd()      # ✗ Enseigné Section 5
```

### Ex 5.6 (Approximation de Rang Faible)
```python
# Utilise :
U, s, Vt = np.linalg.svd(A)  # ✓ Enseigné Section 5
# Pour approximer A ≈ U[:,:k] @ diag(s[:k]) @ Vt[:k,:]
```

### Ex 5.4 (Pseudo-Inverse)
```python
# Implémentation pédagogique :
# 1. Calcule SVD
# 2. Inverse les σᵢ > seuil
# 3. Reconstruit A⁺ = V·Σ⁺·Uᵀ
# 4. Valide contre np.linalg.pinv()
```

---

## ✅ Checklist de Validation

| Correction | Fichier | Statut |
|-----------|---------|--------|
| Ex 3.5 remplacé | exercices_04 | ✓ |
| Ex 5.6 créé | exercices_04 | ✓ |
| Solution 3.5 mise à jour | solutions_04 | ✓ |
| Solution 5.6 ajoutée | solutions_04 | ✓ |
| Ex 5.4 enrichi | exercices_04 | ✓ |
| Solution 5.4 enrichie | solutions_04 | ✓ |
| Cohérence temporelle | Both files | ✓ |
| Concepts documentés | Both files | ✓ |

---

## 📝 Notes pour l'Enseignant

1. **Ordre d'enseignement recommandé:**
   - Section 1-2: Espaces et Indépendance
   - Section 3: Rang (utilise matrix_rank)
   - Section 4: Eigenvalues/Eigenvectors
   - **Section 5: SVD (PUIS Exercices 5.1-5.6)**
   - Section 6: PCA

2. **Points pédagogiques clés:**
   - Ex 3.5 (nouveau): Montre les limites du rang (inégalités)
   - Ex 5.4: Introduit la généralisation des inverses
   - Ex 5.6: Applique SVD pour compression

3. **Relation Ex 5.4 ↔ Ex 5.6:**
   - Ex 5.4: Compute A⁺ via SVD (théorie)
   - Ex 5.6: Use SVD for low-rank approximation (pratique)
   - Cohérent: Tous deux utilisent SVD

---

**Date de correction:** 2025-11-28
**Statut:** ✅ Complet et validé
