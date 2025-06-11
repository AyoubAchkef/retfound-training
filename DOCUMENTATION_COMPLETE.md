<documentation>

# 📚 Documentation Complète - Projet RETFound Training

## 🎯 Introduction

Le projet **RETFound Training** est un système avancé d'intelligence artificielle spécialement conçu pour l'analyse d'images médicales ophtalmologiques. Il utilise des modèles de Vision Transformer (ViT) pré-entraînés pour diagnostiquer automatiquement les maladies rétiniennes à partir d'images de fond d'œil (Fundus) et de tomographie par cohérence optique (OCT).

Ce système peut identifier 28 types différents de conditions oculaires, allant des cas normaux aux pathologies critiques nécessitant une intervention médicale urgente.

---

## 🌳 Structure du Projet

```
retfound_training/
│
├── 📁 configs/                          # Fichiers de configuration
│   ├── 📄 dataset_v6.1.yaml           # Configuration principale pour le dataset v6.1
│   ├── 📄 default.yaml                # Configuration par défaut
│   ├── 📄 local_windows.yaml          # Configuration pour Windows
│   ├── 📄 runpod.yaml                 # Configuration pour RunPod (cloud)
│   ├── 📁 experimental/               # Configurations expérimentales
│   │   ├── 📄 sam_optimizer.yaml      # Optimiseur SAM avancé
│   │   └── 📄 tta_enhanced.yaml       # Augmentation de test améliorée
│   └── 📁 production/                 # Configurations de production
│       ├── 📄 a100_optimized.yaml     # Optimisé pour GPU A100
│       └── 📄 multi_gpu.yaml          # Configuration multi-GPU
│
├── 📁 docs/                            # Documentation du projet
│
├── 📁 retfound/                        # Package principal du système
│   ├── 📄 __init__.py                 # Initialisation du package
│   ├── 📄 __version__.py              # Informations de version
│   │
│   ├── 📁 cli/                        # Interface en ligne de commande
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main.py                 # Point d'entrée principal
│   │   ├── 📄 utils.py                # Utilitaires CLI
│   │   └── 📁 commands/               # Commandes disponibles
│   │       ├── 📄 __init__.py
│   │       ├── 📄 evaluate.py         # Commande d'évaluation
│   │       ├── 📄 export.py           # Commande d'exportation
│   │       ├── 📄 predict.py          # Commande de prédiction
│   │       └── 📄 train.py            # Commande d'entraînement
│   │
│   ├── 📁 core/                       # Composants centraux
│   │   ├── 📄 __init__.py
│   │   ├── 📄 config.py               # Gestion des configurations
│   │   ├── 📄 constants.py            # Constantes globales
│   │   ├── 📄 exceptions.py           # Gestion des erreurs
│   │   └── 📄 registry.py             # Registre des composants
│   │
│   ├── 📁 data/                       # Gestion des données
│   │   ├── 📄 __init__.py
│   │   ├── 📄 cache.py                # Système de cache
│   │   ├── 📄 datamodule.py           # Module de données PyTorch Lightning
│   │   ├── 📄 datasets.py             # Définition des datasets
│   │   ├── 📄 samplers.py             # Échantillonneurs de données
│   │   └── 📄 transforms.py           # Transformations d'images
│   │
│   ├── 📁 evaluation/                 # Système d'évaluation
│   │   └── 📄 evaluator.py            # Évaluateur principal
│   │
│   ├── 📁 export/                     # Exportation de modèles
│   │   ├── 📄 __init__.py
│   │   ├── 📄 exporter.py             # Exportateur principal
│   │   ├── 📄 inference.py            # Moteur d'inférence
│   │   ├── 📄 onnx.py                 # Export ONNX
│   │   ├── 📄 tensortrt.py            # Export TensorRT
│   │   └── 📄 torchscript.py          # Export TorchScript
│   │
│   ├── 📁 metrics/                    # Métriques médicales
│   │   ├── 📄 __init__.py
│   │   ├── 📄 medical.py              # Métriques spécialisées
│   │   ├── 📄 report.py               # Génération de rapports
│   │   └── 📄 visualization.py        # Visualisations
│   │
│   ├── 📁 models/                     # Architectures de modèles
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base.py                 # Modèle de base
│   │   ├── 📄 factory.py              # Fabrique de modèles
│   │   ├── 📄 retfound.py             # Modèle RETFound
│   │   └── 📁 layers/                 # Couches personnalisées
│   │       ├── 📄 __init__.py
│   │       ├── 📄 attention.py        # Mécanismes d'attention
│   │       ├── 📄 blocks.py           # Blocs de construction
│   │       └── 📄 embeddings.py       # Embeddings
│   │
│   ├── 📁 training/                   # Système d'entraînement
│   │   ├── 📄 __init__.py
│   │   ├── 📄 losses.py               # Fonctions de perte
│   │   ├── 📄 optimizers.py           # Optimiseurs
│   │   ├── 📄 schedulers.py           # Planificateurs de taux d'apprentissage
│   │   ├── 📄 trainer.py              # Entraîneur principal
│   │   ├── 📁 callbacks/              # Callbacks d'entraînement
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 base.py             # Callback de base
│   │   │   ├── 📄 checkpoint.py       # Sauvegarde de checkpoints
│   │   │   ├── 📄 early_stopping.py   # Arrêt précoce
│   │   │   ├── 📄 logging.py          # Journalisation
│   │   │   └── 📄 metrics.py          # Suivi des métriques
│   │   └── 📁 strategies/             # Stratégies d'entraînement
│   │       ├── 📄 __init__.py
│   │       ├── 📄 base.py             # Stratégie de base
│   │       ├── 📄 ddp.py              # Parallélisation distribuée
│   │       └── 📄 single_gpu.py       # GPU unique
│   │
│   └── 📁 utils/                      # Utilitaires généraux
│       ├── 📄 __init__.py
│       ├── 📄 device.py               # Gestion des périphériques
│       ├── 📄 io.py                   # Entrées/sorties
│       ├── 📄 logging.py              # Système de logs
│       ├── 📄 reproducibility.py      # Reproductibilité
│       └── 📄 timing.py               # Mesure du temps
│
├── 📁 scripts/                        # Scripts utilitaires
│   ├── 📄 benchmark.py                # Tests de performance
│   ├── 📄 download_weights.py         # Téléchargement des poids
│   ├── 📄 setup_environment.sh        # Configuration de l'environnement
│   └── 📄 validate_dataset_v61.py     # Validation du dataset v6.1
│
├── 📁 tests/                          # Tests automatisés
│   ├── 📄 __init__.py
│   ├── 📄 conftest.py                 # Configuration des tests
│   ├── 📁 fixtures/                   # Données de test
│   │   └── 📄 __init__.py
│   ├── 📁 integration/                # Tests d'intégration
│   │   ├── 📄 __init__.py
│   │   ├── 📄 test_cli.py             # Tests CLI
│   │   ├── 📄 test_export.py          # Tests d'exportation
│   │   └── 📄 test_training.py        # Tests d'entraînement
│   └── 📁 unit/                       # Tests unitaires
│       ├── 📄 __init__.py
│       ├── 📄 test_config.py          # Tests de configuration
│       ├── 📄 test_datasets.py        # Tests des datasets
│       ├── 📄 test_losses.py          # Tests des fonctions de perte
│       ├── 📄 test_metrics.py         # Tests des métriques
│       ├── 📄 test_models.py          # Tests des modèles
│       ├── 📄 test_optimizers.py      # Tests des optimiseurs
│       └── 📄 test_utils.py           # Tests des utilitaires
│
├── 📁 weights/                        # Poids pré-entraînés
│
├── 📄 .env.example                    # Exemple de variables d'environnement
├── 📄 .gitignore                      # Fichiers ignorés par Git
├── 📄 architecture projet caasi retfound.txt  # Architecture du projet
├── 📄 CORRECTIONS_APPLIED_V61.md      # Corrections appliquées v6.1
├── 📄 demo_v61_usage.py               # Démonstration d'utilisation v6.1
├── 📄 LICENSE                         # Licence du projet
├── 📄 MIGRATION_V61_COMPLETE.md       # Guide de migration v6.1
├── 📄 pyproject.toml                  # Configuration du projet Python
├── 📄 README.md                       # Documentation principale
├── 📄 setup.cfg                       # Configuration de setup
├── 📄 test_basic_structure.py         # Test de structure de base
└── 📄 test_v61_setup.py               # Test de configuration v6.1
```

---

## 📋 Description Détaillée des Composants

### 🔧 Fichiers de Configuration Principaux

#### **pyproject.toml**
Fichier de configuration principal du projet Python utilisant Poetry. Il définit :
- Les dépendances du projet (PyTorch, timm, albumentations, etc.)
- Les métadonnées du projet (nom, version, auteurs)
- Les outils de développement (tests, formatage, linting)
- Les scripts en ligne de commande disponibles

#### **README.md**
Documentation principale du projet contenant :
- Instructions d'installation et de configuration
- Guide de démarrage rapide
- Exemples d'utilisation des commandes
- Description des fonctionnalités principales

#### **.env.example**
Modèle de fichier d'environnement définissant les variables nécessaires :
- Chemins vers les datasets
- Clés API pour les services de monitoring
- Configurations spécifiques à l'environnement

### 📁 Dossier `configs/`

Ce dossier contient tous les fichiers de configuration YAML qui définissent les paramètres d'entraînement et d'évaluation.

#### **dataset_v6.1.yaml**
Configuration principale pour le dataset CAASI v6.1 :
- **Fonction** : Définit tous les paramètres pour travailler avec 211,952 images médicales
- **Classes supportées** : 28 classes (18 Fundus + 10 OCT)
- **Optimisations** : Configuration pour GPU A100, SAM optimizer, EMA
- **Augmentation** : MixUp, CutMix, RandAugment pour améliorer la robustesse

#### **default.yaml**
Configuration de base utilisée comme point de départ pour tous les autres fichiers de configuration.

#### **local_windows.yaml**
Configuration adaptée pour les systèmes Windows, avec des ajustements pour :
- Chemins de fichiers Windows
- Gestion des processus multiples
- Optimisations spécifiques à Windows

#### **runpod.yaml**
Configuration optimisée pour l'entraînement sur la plateforme cloud RunPod.

### 📁 Dossier `retfound/`

Package principal contenant toute la logique du système d'IA médicale.

#### 📁 Sous-dossier `cli/`

Interface en ligne de commande permettant d'utiliser le système facilement.

**main.py** : Point d'entrée principal qui permet d'exécuter des commandes comme :
```bash
retfound train --config configs/dataset_v6.1.yaml
retfound predict image.jpg --model checkpoints/best.pth
```

**commands/** : Contient les différentes commandes disponibles :
- **train.py** : Lance l'entraînement d'un modèle
- **evaluate.py** : Évalue les performances d'un modèle
- **predict.py** : Fait des prédictions sur de nouvelles images
- **export.py** : Exporte un modèle vers différents formats

#### 📁 Sous-dossier `core/`

Composants centraux du système.

**constants.py** : Définit toutes les constantes importantes :
- Noms des 28 classes de maladies
- Statistiques du dataset v6.1 (211,952 images)
- Conditions critiques nécessitant une attention médicale urgente
- Paramètres des modèles Vision Transformer

**config.py** : Gère le chargement et la validation des fichiers de configuration.

**exceptions.py** : Définit les erreurs personnalisées du système.

#### 📁 Sous-dossier `data/`

Gestion complète des données médicales.

**datasets.py** : Définit comment charger et traiter les images médicales :
- Support des images Fundus (fond d'œil) et OCT
- Gestion des 28 classes de pathologies
- Équilibrage automatique des classes

**transforms.py** : Transformations spécialisées pour les images médicales :
- Normalisation adaptée aux images rétiniennes
- Augmentations préservant les caractéristiques médicales importantes

**cache.py** : Système de cache pour accélérer le chargement des données.

#### 📁 Sous-dossier `models/`

Architectures des modèles d'intelligence artificielle.

**retfound.py** : Implémentation du modèle RETFound :
- Vision Transformer Large avec 632 millions de paramètres
- Pré-entraîné sur 1,6 million d'images rétiniennes
- Adapté pour la classification de 28 pathologies

**factory.py** : Fabrique permettant de créer différents types de modèles.

**layers/** : Couches personnalisées pour les Vision Transformers :
- **attention.py** : Mécanismes d'attention multi-têtes
- **blocks.py** : Blocs de construction des transformers
- **embeddings.py** : Embeddings de position et de patch

#### 📁 Sous-dossier `training/`

Système d'entraînement avancé.

**trainer.py** : Entraîneur principal gérant :
- L'entraînement sur GPU A100
- La validation en temps réel
- La sauvegarde automatique des checkpoints

**optimizers.py** : Optimiseurs spécialisés :
- SAM (Sharpness Aware Minimization) pour une meilleure généralisation
- AdamW avec décroissance par couches pour les Vision Transformers

**losses.py** : Fonctions de perte adaptées au médical :
- Cross-entropy avec lissage d'étiquettes
- Focal loss pour les classes déséquilibrées

**callbacks/** : Callbacks pour surveiller l'entraînement :
- **early_stopping.py** : Arrêt précoce si pas d'amélioration
- **checkpoint.py** : Sauvegarde automatique des meilleurs modèles
- **metrics.py** : Calcul des métriques médicales en temps réel

#### 📁 Sous-dossier `metrics/`

Métriques spécialisées pour l'évaluation médicale.

**medical.py** : Métriques adaptées à l'ophtalmologie :
- Sensibilité et spécificité par classe
- Cohen's Kappa pour l'accord inter-observateurs
- AUC-ROC macro et pondérée
- Surveillance spéciale des conditions critiques

**report.py** : Génération de rapports cliniques détaillés.

**visualization.py** : Création de graphiques et visualisations pour l'analyse des résultats.

#### 📁 Sous-dossier `export/`

Exportation des modèles vers différents formats.

**exporter.py** : Exportateur principal supportant :
- ONNX pour l'interopérabilité
- TorchScript pour la production
- TensorRT pour l'optimisation GPU

**inference.py** : Moteur d'inférence optimisé pour la production.

### 📁 Dossier `scripts/`

Scripts utilitaires pour faciliter l'utilisation du système.

**download_weights.py** : Télécharge automatiquement les poids pré-entraînés RETFound depuis GitHub.

**validate_dataset_v61.py** : Valide la structure et l'intégrité du dataset v6.1 :
- Vérifie la présence des 28 classes
- Contrôle la répartition train/val/test
- Valide le format des images

**benchmark.py** : Teste les performances du système sur différentes configurations matérielles.

### 📁 Dossier `tests/`

Suite complète de tests automatisés.

**unit/** : Tests unitaires pour chaque composant :
- **test_models.py** : Teste les architectures de modèles
- **test_datasets.py** : Teste le chargement des données
- **test_metrics.py** : Teste les calculs de métriques

**integration/** : Tests d'intégration :
- **test_training.py** : Teste l'entraînement complet
- **test_cli.py** : Teste les commandes en ligne de commande

### 📁 Dossier `weights/`

Stockage des poids pré-entraînés :
- **RETFound_mae_natureCFP.pth** : Poids pour images Fundus
- **RETFound_mae_natureOCT.pth** : Poids pour images OCT

---

## 🎯 Fonctionnalités Principales

### 🔬 **Analyse Médicale Avancée**
- **28 Classes de Pathologies** : Diagnostic automatique de conditions normales et pathologiques
- **Modalités Multiples** : Support des images Fundus et OCT
- **Conditions Critiques** : Surveillance spéciale des urgences médicales (RAO, RVO, décollement rétinien)

### 🚀 **Performance Optimisée**
- **GPU A100** : Configuration optimisée pour les derniers GPU NVIDIA
- **Précision Mixed** : Utilisation de BFloat16 pour accélérer l'entraînement
- **Compilation PyTorch** : Optimisations automatiques avec torch.compile

### 📊 **Monitoring Complet**
- **TensorBoard** : Visualisation en temps réel des métriques
- **Weights & Biases** : Suivi avancé des expériences
- **Rapports Cliniques** : Génération automatique de rapports médicaux

### 🔧 **Facilité d'Utilisation**
- **Interface CLI** : Commandes simples pour toutes les opérations
- **Configuration YAML** : Paramétrage facile via fichiers de configuration
- **Export Multiple** : Support ONNX, TorchScript, TensorRT

---

## 📈 Dataset v6.1 - Caractéristiques

Le système utilise le dataset CAASI v6.1, une collection massive d'images médicales ophtalmologiques :

### 📊 **Statistiques Globales**
- **Total d'images** : 211,952
- **Images Fundus** : 44,815 (21.1%)
- **Images OCT** : 167,137 (78.9%)
- **Répartition** : 80% entraînement / 10% validation / 10% test

### 🏥 **Classes Médicales**

#### **Images Fundus (18 classes)**
1. Normal_Fundus - Fond d'œil normal
2. DR_Mild - Rétinopathie diabétique légère
3. DR_Moderate - Rétinopathie diabétique modérée
4. DR_Severe - Rétinopathie diabétique sévère
5. DR_Proliferative - Rétinopathie diabétique proliférative
6. Glaucoma_Suspect - Suspicion de glaucome
7. Glaucoma_Positive - Glaucome confirmé
8. RVO - Occlusion veineuse rétinienne
9. RAO - Occlusion artérielle rétinienne
10. Hypertensive_Retinopathy - Rétinopathie hypertensive
11. Drusen - Drusen maculaires
12. CNV_Wet_AMD - DMLA humide avec néovascularisation
13. Myopia_Degenerative - Myopie dégénérative
14. Retinal_Detachment - Décollement rétinien
15. Macular_Scar - Cicatrice maculaire
16. Cataract_Suspected - Suspicion de cataracte
17. Optic_Disc_Anomaly - Anomalie du disque optique
18. Other - Autres pathologies

#### **Images OCT (10 classes)**
1. Normal_OCT - OCT normal
2. DME - Œdème maculaire diabétique
3. CNV_OCT - Néovascularisation choroïdienne
4. Dry_AMD - DMLA sèche
5. ERM - Membrane épirétinienne
6. Vitreomacular_Interface_Disease - Maladie de l'interface vitréo-maculaire
7. CSR - Choriorétinopathie séreuse centrale
8. RVO_OCT - Occlusion veineuse (OCT)
9. Glaucoma_OCT - Glaucome (OCT)
10. RAO_OCT - Occlusion artérielle (OCT)

---

## 🚨 Conditions Critiques Surveillées

Le système surveille automatiquement les conditions nécessitant une attention médicale urgente :

### 🔴 **Urgences Absolues**
- **RAO (Occlusion Artérielle Rétinienne)** : Seuil de sensibilité minimum 99%
- **Décollement Rétinien** : Seuil de sensibilité minimum 99%

### 🟡 **Conditions Urgentes**
- **RVO (Occlusion Veineuse Rétinienne)** : Seuil de sensibilité minimum 97%
- **CNV (Néovascularisation Choroïdienne)** : Seuil de sensibilité minimum 98%
- **Rétinopathie Diabétique Proliférative** : Seuil de sensibilité minimum 98%

### 🟠 **Surveillance Renforcée**
- **Œdème Maculaire Diabétique (DME)** : Seuil de sensibilité minimum 95%
- **Glaucome Confirmé** : Seuil de sensibilité minimum 95%

---

## 🎯 Résumé et Objectif Global

Le projet **RETFound Training** représente une solution complète et avancée pour l'analyse automatisée d'images médicales ophtalmologiques. Il combine :

### 🔬 **Excellence Scientifique**
- Modèles Vision Transformer de pointe avec 632M de paramètres
- Pré-entraînement sur 1,6 million d'images rétiniennes
- Métriques médicales spécialisées et validation clinique

### 🏥 **Impact Médical**
- Diagnostic automatique de 28 pathologies oculaires
- Surveillance des conditions critiques nécessitant une intervention urgente
- Support pour les deux principales modalités d'imagerie (Fundus et OCT)

### 🚀 **Performance Technique**
- Optimisations pour GPU A100 et infrastructure cloud
- Système de cache intelligent et chargement de données optimisé
- Export vers multiples formats pour déploiement en production

### 🔧 **Facilité d'Utilisation**
- Interface en ligne de commande intuitive
- Configuration flexible via fichiers YAML
- Documentation complète et exemples d'utilisation

Ce système vise à assister les professionnels de santé dans le diagnostic précoce et précis des maladies rétiniennes, contribuant ainsi à la préservation de la vision et à l'amélioration des soins ophtalmologiques.

</documentation>
