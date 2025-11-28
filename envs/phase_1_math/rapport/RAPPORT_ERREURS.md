# Rapport de Vérification des Notebooks de Solutions

Date: 28 Novembre 2025
Environnement: `llm` (Miniconda)

## ✅ Notebooks Valides
Les notebooks suivants s'exécutent sans erreur :
- `solutions_03_algebre_fondamentale.ipynb`
- `solutions_04_algebre_avancee.ipynb`

## ❌ Notebooks avec Erreurs
Voici les erreurs détectées à corriger :

### 1. `solutions_05_calcul_differentiel.ipynb`
**Erreur :** `SyntaxError: f-string expression part cannot include a backslash`
**Détail :**
```python
print(f'{"h":>12} | {"f\'(x) approx":>15} | {"Erreur":>12}')
```
**Piste de correction :** Les backslashs `\` ne sont pas autorisés dans les expressions `{}` des f-strings en Python < 3.12 (ou selon le contexte). Il faut sortir les guillemets ou utiliser une autre méthode de formatage.

### 2. `solutions_06_calcul_integral.ipynb`
**Erreur :** `SyntaxError: invalid character '∞' (U+221E)`
**Détail :**
```python
(-∞)
```
**Piste de correction :** Le symbole `∞` a été utilisé directement dans le code (probablement copier-coller de texte mathématique). Il faut le remplacer par `float('inf')` ou `np.inf` ou le mettre en commentaire si c'est du texte.

### 3. `solutions_07_probabilites.ipynb`
**Erreur :** `SyntaxError: unexpected character after line continuation character`
**Détail :**
```python
print('SOLUTION 6.2: PROPRIÉTÉS DE L'ESPÉRANCE\n' + '='*60)
```
**Piste de correction :** Il y a une apostrophe non échappée dans `L'ESPÉRANCE` à l'intérieur d'une chaîne délimitée par des apostrophes `'...'`. Utiliser des guillemets doubles `print("...")` ou échapper l'apostrophe `L\'ESPÉRANCE`.

### 4. `solutions_08_statistiques.ipynb`
**Erreur :** `SyntaxError: unterminated string literal`
**Détail :**
```text
The TodoWrite tool hasn't been used recently...
```
**Piste de correction :** Il semble qu'un message système ou un log d'outil IA ait été accidentellement collé dans une cellule de code. Il faut supprimer ce texte parasite.
