#!/usr/bin/env python3
"""
Script de correction DIRECTE pour datasets.py - Approche ultra-simple
"""

def fix_datasets_direct():
    """Correction DIRECTE du fichier datasets.py"""
    
    file_path = "retfound/data/datasets.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📁 Lecture du fichier {file_path}")
        
        # Correction DIRECTE des patterns problématiques avec indentation correcte
        
        # Pattern 1: Corriger les blocs de transformation avec indentation correcte
        old_pattern1 = """if hasattr(self.transform, 'transform'):  # Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
        else:  # torchvision
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = self.transform(image)"""
        
        new_pattern1 = """        if hasattr(self.transform, 'transform'):  # Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
        else:  # torchvision
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = self.transform(image)"""
        
        # Pattern 2: Corriger les blocs de fromarray avec indentation correcte
        old_pattern2 = """if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))"""
        
        new_pattern2 = """        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))"""
        
        # Appliquer les corrections
        count1 = content.count(old_pattern1)
        content = content.replace(old_pattern1, new_pattern1)
        
        count2 = content.count(old_pattern2)
        content = content.replace(old_pattern2, new_pattern2)
        
        # S'assurer que numpy est importé
        if "import numpy as np" not in content:
            content = content.replace("from PIL import Image", "import numpy as np\nfrom PIL import Image", 1)
        
        # Sauvegarder
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        total_changes = count1 + count2
        print(f"✅ Corrections d'indentation effectuées:")
        print(f"   - {count1} blocs de transformation corrigés")
        print(f"   - {count2} blocs fromarray corrigés")
        print(f"   - Total: {total_changes} corrections")
        
        if total_changes > 0:
            print("🔄 Redémarrez le test: python test_training_step_by_step.py")
        else:
            print("ℹ️  Aucune correction nécessaire")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("🔧 Correction DIRECTE d'indentation de datasets.py...")
    fix_datasets_direct()
    print("✅ Correction terminée")
