#!/usr/bin/env python3
"""
Script de correction complète pour tous les problèmes Image.fromarray sur RunPod
"""

import re

def fix_all_image_issues():
    """Corrige TOUS les problèmes d'images dans datasets.py"""
    
    file_path = "retfound/data/datasets.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📁 Lecture du fichier {file_path}")
        
        changes_made = 0
        
        # 1. Corriger TOUS les appels Image.fromarray(image) sans vérification
        pattern1 = r'(\s+)image = Image\.fromarray\(image\)'
        if re.search(pattern1, content):
            content = re.sub(
                pattern1,
                r'''\1if isinstance(image, np.ndarray):
\1    image = Image.fromarray(image)
\1elif not isinstance(image, Image.Image):
\1    image = Image.fromarray(np.array(image))''',
                content
            )
            changes_made += 1
            print("✅ Correction 1: Tous les Image.fromarray corrigés")
        
        # 2. Corriger les appels self.transform(image) directs
        pattern2 = r'(\s+)image = self\.transform\(image\)(?!\s*\n\s*if)'
        if re.search(pattern2, content):
            content = re.sub(
                pattern2,
                r'''\1if hasattr(self.transform, 'transform'):  # Albumentations
\1    augmented = self.transform(image=image)
\1    image = augmented['image']
\1else:  # torchvision
\1    if isinstance(image, np.ndarray):
\1        image = Image.fromarray(image)
\1    elif not isinstance(image, Image.Image):
\1        image = Image.fromarray(np.array(image))
\1    image = self.transform(image)''',
                content
            )
            changes_made += 1
            print("✅ Correction 2: Tous les self.transform corrigés")
        
        # 3. Ajouter les imports nécessaires si manquants
        if 'import numpy as np' not in content:
            content = content.replace('import numpy as np', '', 1)  # Remove if exists
            content = content.replace('from PIL import Image', 'import numpy as np\nfrom PIL import Image', 1)
            changes_made += 1
            print("✅ Correction 3: Import numpy ajouté")
        
        # 4. Vérification ligne par ligne pour les cas manqués
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Chercher les lignes problématiques
            if 'Image.fromarray(image)' in line and 'isinstance' not in line:
                print(f"⚠️  Ligne problématique trouvée à la ligne {i+1}: {line.strip()}")
                indent = len(line) - len(line.lstrip())
                replacement_lines = [
                    ' ' * indent + 'if isinstance(image, np.ndarray):',
                    ' ' * indent + '    image = Image.fromarray(image)',
                    ' ' * indent + 'elif not isinstance(image, Image.Image):',
                    ' ' * indent + '    image = Image.fromarray(np.array(image))'
                ]
                lines[i] = '\n'.join(replacement_lines)
                changes_made += 1
                print(f"✅ Ligne {i+1} corrigée")
            
            # Chercher les transform directs
            elif 'image = self.transform(image)' in line and 'hasattr' not in lines[max(0, i-2):i+1]:
                print(f"⚠️  Transform direct trouvé à la ligne {i+1}: {line.strip()}")
                indent = len(line) - len(line.lstrip())
                replacement_lines = [
                    ' ' * indent + 'if hasattr(self.transform, \'transform\'):  # Albumentations',
                    ' ' * indent + '    augmented = self.transform(image=image)',
                    ' ' * indent + '    image = augmented[\'image\']',
                    ' ' * indent + 'else:  # torchvision',
                    ' ' * indent + '    if isinstance(image, np.ndarray):',
                    ' ' * indent + '        image = Image.fromarray(image)',
                    ' ' * indent + '    elif not isinstance(image, Image.Image):',
                    ' ' * indent + '        image = Image.fromarray(np.array(image))',
                    ' ' * indent + '    image = self.transform(image)'
                ]
                lines[i] = '\n'.join(replacement_lines)
                changes_made += 1
                print(f"✅ Transform ligne {i+1} corrigé")
        
        if changes_made > 0:
            content = '\n'.join(lines)
            
            # Sauvegarder le fichier corrigé
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Fichier {file_path} corrigé avec {changes_made} changements")
            print("🔄 Redémarrez le test: python test_training_step_by_step.py")
        else:
            print("ℹ️  Aucune correction nécessaire trouvée")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("🔧 Correction COMPLÈTE de tous les problèmes d'images...")
    fix_all_image_issues()
    print("✅ Correction terminée")
