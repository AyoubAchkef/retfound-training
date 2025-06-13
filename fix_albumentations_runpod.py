#!/usr/bin/env python3
"""
Script de correction spécifique pour les transformations Albumentations sur RunPod
"""

import re

def fix_albumentations_transforms():
    """Corrige les transformations Albumentations dans datasets.py"""
    
    file_path = "retfound/data/datasets.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📁 Lecture du fichier {file_path}")
        
        # Pattern pour trouver les appels directs à self.transform(image)
        # qui ne sont pas déjà dans un bloc conditionnel correct
        
        # Remplacement 1: Dans BaseDataset.__getitem__
        pattern1 = r'(\s+)# Apply transforms\s*\n(\s+)if self\.transform is not None:\s*\n(\s+)image = self\.transform\(image\)'
        replacement1 = r'''\1# Apply transforms
\2if self.transform is not None:
\2    # Handle different transform types
\2    if hasattr(self.transform, 'transform'):  # Albumentations
\2        augmented = self.transform(image=image)
\2        image = augmented['image']
\2    else:  # torchvision
\2        image = Image.fromarray(image)
\2        image = self.transform(image)'''
        
        # Remplacement 2: Dans CAASIDatasetV61.__getitem__
        pattern2 = r'(\s+)# Apply transforms\s*\n(\s+)if self\.transform is not None:\s*\n(\s+)# Handle different transform types\s*\n(\s+)if hasattr\(self\.transform, \'transform\'\):.*?\n(\s+)augmented = self\.transform\(image=image\)\s*\n(\s+)image = augmented\[\'image\'\]\s*\n(\s+)else:.*?\n(\s+)image = Image\.fromarray\(image\)\s*\n(\s+)image = self\.transform\(image\)'
        
        # Chercher et remplacer tous les appels directs restants
        patterns_to_fix = [
            (r'(\s+)image = self\.transform\(image\)(?!\s*\n\s*else)', 
             r'''\1# Handle different transform types
\1if hasattr(self.transform, 'transform'):  # Albumentations
\1    augmented = self.transform(image=image)
\1    image = augmented['image']
\1else:  # torchvision
\1    image = Image.fromarray(image)
\1    image = self.transform(image)''')
        ]
        
        changes_made = 0
        
        for pattern, replacement in patterns_to_fix:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made += 1
                print(f"✅ Correction appliquée: pattern {changes_made}")
        
        # Vérification spéciale pour les lignes problématiques
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'image = self.transform(image)' in line and 'hasattr' not in lines[max(0, i-3):i+1]:
                print(f"⚠️  Ligne problématique trouvée à la ligne {i+1}: {line.strip()}")
                # Remplacer cette ligne spécifique
                indent = len(line) - len(line.lstrip())
                replacement_lines = [
                    ' ' * indent + '# Handle different transform types',
                    ' ' * indent + 'if hasattr(self.transform, \'transform\'):  # Albumentations',
                    ' ' * indent + '    augmented = self.transform(image=image)',
                    ' ' * indent + '    image = augmented[\'image\']',
                    ' ' * indent + 'else:  # torchvision',
                    ' ' * indent + '    image = Image.fromarray(image)',
                    ' ' * indent + '    image = self.transform(image)'
                ]
                lines[i] = '\n'.join(replacement_lines)
                changes_made += 1
                print(f"✅ Ligne {i+1} corrigée")
        
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
    print("🔧 Correction des transformations Albumentations...")
    fix_albumentations_transforms()
    print("✅ Correction terminée")
