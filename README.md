

# Vision Par Ordinateur Lab : Invariance, Robustesse et Analyse d'Intensité 🎓

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV-5C3EE8.svg)](https://opencv.org/)

Ce dépôt contient le code source et les fondements théoriques d'un laboratoire interactif dédié à **l'analyse de l'intensité et des histogrammes**, développé dans le cadre du cours *INF 4238 : Vision par Ordinateur* à l'**Université de Yaoundé I**.

L'objectif principal est d'étudier et de quantifier l'impact des techniques d'égalisation et de normalisation sur la robustesse et l'invariance des descripteurs globaux d'images face aux variations d'éclairage.

---

## 📑 Table des Matières
- [Problématique Centrale](#-problématique-centrale)
- [Fonctionnalités & Pipeline](#-fonctionnalités--pipeline)
- [Méthodes de Prétraitement Implémentées](#-méthodes-de-prétraitement-implémentées)
- [Descripteurs Globaux Étudiés](#-descripteurs-globaux-étudiés)
- [Architecture du Code](#-architecture-du-code)
- [Installation et Lancement](#-installation-et-lancement)
- [Principaux Résultats Synthétisés](#-principaux-résultats-synthétisés)


---

## 🎯 Problématique Centrale
Une même scène photographiée sous des conditions lumineuses changeantes produit des matrices de pixels aux valeurs numériques divergentes. Comment extraire des caractéristiques visuelles (descripteurs) qui restent **invariantes aux conditions d'acquisition** tout en demeurant **discriminantes** pour distinguer des scènes différentes?

---

## 🚀 Fonctionnalités & Pipeline
L'application fournit une interface web interactive (Streamlit) permettant de suivre un pipeline complet en 4 étapes:
1. **Simulation de stress lumineux (Scientific Stress Kernel)** : Assombrissement (*Low Light*), Surexposition (*Burnout*), ou Faible Contraste (*Brouillard*).
2. **Harmonisation & Égalisation** : Application des filtres de correction de l'intensité (Min-Max, GHE, CLAHE).
3. **Extraction de signatures graphiques** : Calcul en temps réel des descripteurs sur l'image originale et dégradée.
4. **Benchmarking Quantitatif** : Calcul automatique de la **similarité cosinus** pour mesurer le gain d'invariance apporté par le prétraitement.

---

## 🧪 Méthodes de Prétraitement Implémentées
- **Égalisation Globale d'Histogramme (GHE)** : Redistribution uniforme des niveaux d'intensité basée sur la fonction de répartition cumulative (CDF) globale.
- **Égalisation Adaptative Limitée par Contraste (CLAHE)** : Approche spatio-contextuelle découpant l'image en tuiles ($8\times8$) avec écrêtage de l'histogramme local pour limiter l'amplification du bruit.
- **Normalisation Min-Max** : Recadrage dynamique linéaire des intensités dans l'intervalle $[0, 255]$.

---

## 🧬 Descripteurs Globaux Étudiés
L'application extrait et compare trois types majeurs de signatures:
- **Histogramme de Couleurs** : Capture la distribution globale des teintes (3 canaux RGB).
- **Local Binary Patterns (LBP)** : Descripteur de texture locale, naturellement invariant aux transformations d'intensité monotones.
- **Histogramme des Gradients Orientés (HOG)** : Analyse des formes et contours via la quantification des orientations de gradients.

La robustesse est mesurée à l'aide de la **similarité cosinus** entre les vecteurs caractéristiques[cite: 433]:
$$\text{Similarité}(\phi_1, \phi_2) = \frac{\phi_1^\top \phi_2}{\|\phi_1\|_2 \|\phi_2\|_2} \times 100$$

---

## 📂 Architecture du Code
- `app.py` : Script principal contenant l'interface utilisateur Streamlit , le noyau de traitement des images (`scikit-image`, `OpenCV`)  et le module d'affichage des graphiques (`matplotlib`).
- `Rapport.pdf` : Rapport académique complet détaillant les démonstrations et résultats.

---

## ⚙️ Installation et Lancement

### Prerequis
Assurez-vous d'avoir Python 3.8 ou une version supérieure installée sur votre machine.

### 1. Cloner le dépôt
```bash
git clone  https://github.com/lethyciamelong/Analyse-Intensite-Histogrammes.git
cd Analyse-Intensite-Histogrammes

```




### 2. Installer les dépendances

Installez toutes les bibliothèques requises à partir du fichier `requirements.txt` :

```bash
pip install -r requirements.txt

```


### 3. Exécuter l'application Streamlit

Lancez l'interface interactive locale avec la commande suivante :

```bash
streamlit run app.py

```

---

## 📊 Principaux Résultats Synthétisés

* **CLAHE** surclasse systématiquement la méthode globale GHE en évitant l'effet de *wash-out* (délavage des couleurs) et en préservant de manière optimale les textures locales.
* **HOG + CLAHE** produit le couple le plus stable face aux variations d'illumination non-linéaires grâce à la normalisation intrinsèque par blocs du descripteur HOG.
* La **normalisation $L^2$** sur les vecteurs de descripteurs offre une robustesse mathématique parfaite ($100\%$ d'invariance) face aux variations d'éclairage strictement linéaires.

