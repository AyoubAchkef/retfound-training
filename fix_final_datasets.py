#!/usr/bin/env python3
"""
Script de correction FINAL et BRUTAL pour datasets.py sur RunPod
"""

def fix_datasets_final():
    """Correction FINALE et BRUTALE de datasets.py"""
    
    file_path = "retfound/data/datasets.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📁 Lecture du fichier {file_path}")
        
        # Remplacement BRUTAL de TOUS les patterns problématiques
        
        # 1. Remplacer TOUS les "image = self.transform(image)" par la version sécurisée
        old_transform = "image = self.transform(image)"
        new_transform = """if hasattr(self.transform, 'transform'):  # Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
        else:  # torchvision
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = self.transform(image)"""
        
        count1 = content.count(old_transform)
        content = content.replace(old_transform, new_transform)
        
        # 2. Remplacer TOUS les "image = Image.fromarray(image)" par la version sécurisée
        old_fromarray = "image = Image.fromarray(image)"
        new_fromarray = """if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))"""
        
        count2 = content.count(old_fromarray)
        content = content.replace(old_fromarray, new_fromarray)
        
        # 3. S'assurer que numpy est importé
        if "import numpy as np" not in content:
            content = content.replace("from PIL import Image", "import numpy as np\nfrom PIL import Image", 1)
        
        # Sauvegarder
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        total_changes = count1 + count2
        print(f"✅ Remplacements effectués:")
        print(f"   - {count1} appels self.transform(image) corrigés")
        print(f"   - {count2} appels Image.fromarray(image) corrigés")
        print(f"   - Total: {total_changes} corrections")
        
        if total_changes > 0:
            print("🔄 Redémarrez le test: python test_training_step_by_step.py")
        else:
            print("ℹ️  Aucune correction nécessaire")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("🔧 Correction FINALE et BRUTALE de datasets.py...")
    fix_datasets_final()
    print("✅ Correction terminée")
