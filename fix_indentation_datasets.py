#!/usr/bin/env python3
"""
Script de correction d'indentation pour datasets.py sur RunPod
"""

def fix_indentation():
    """Corrige les problèmes d'indentation dans datasets.py"""
    
    file_path = "retfound/data/datasets.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📁 Lecture du fichier {file_path}")
        
        # Corriger les problèmes d'indentation spécifiques
        
        # Pattern 1: Corriger l'indentation après les if statements
        content = content.replace(
            """if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))""",
            """        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))"""
        )
        
        # Pattern 2: Corriger l'indentation des blocs de transformation
        content = content.replace(
            """if hasattr(self.transform, 'transform'):  # Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
        else:  # torchvision
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = self.transform(image)""",
            """        if hasattr(self.transform, 'transform'):  # Albumentations
            augmented = self.transform(image=image)
            image = augmented['image']
        else:  # torchvision
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = self.transform(image)"""
        )
        
        # Correction plus agressive ligne par ligne
        lines = content.split('\n')
        corrected_lines = []
        
        for i, line in enumerate(lines):
            # Si on trouve une ligne avec un if/elif/else mal indenté après un bloc de code
            if i > 0 and line.strip().startswith(('if isinstance', 'elif not isinstance', 'else:')):
                # Vérifier l'indentation de la ligne précédente
                prev_line = lines[i-1] if i > 0 else ""
                if prev_line.strip() and not prev_line.startswith('        '):
                    # Ajouter l'indentation appropriée
                    if not line.startswith('        '):
                        line = '        ' + line.lstrip()
            
            # Corriger les lignes qui suivent les if/elif/else
            elif i > 0 and lines[i-1].strip().startswith(('if isinstance', 'elif not isinstance', 'else:')):
                if line.strip() and not line.startswith('            '):
                    line = '            ' + line.lstrip()
            
            corrected_lines.append(line)
        
        content = '\n'.join(corrected_lines)
        
        # Sauvegarder
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Indentation corrigée")
        print("🔄 Redémarrez le test: python test_training_step_by_step.py")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("🔧 Correction d'indentation de datasets.py...")
    fix_indentation()
    print("✅ Correction terminée")
