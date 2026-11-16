import os

def print_clean_tree():
    allowed_top_folders = {'data', 'evaluation', 'latex', 'src', 'tests'}
    latex_ignore_ext = {'.aux', '.log', '.toc', '.bfl', '.bbl', '.bcf', '.blg', '.fdb_latexmk', '.fls', '.idx', '.ilg', '.ind', '.lof', '.lol', '.lot', '.run.xml', '.synctex.gz'}

    print("thesis-root/")
    for item in sorted(os.listdir('.')):
        if item in allowed_top_folders:
            print(f"├── {item}/")
            for root, dirs, files in os.walk(item):
                if '.git' in root or '__pycache__' in root:
                    continue
                    
                level = root.split(os.sep)
                indent = '│   ' * len(level)
                
                if len(level) > 1:
                    print(f"{indent[:-4]}├── {os.path.basename(root)}/")
                
                subindent = '│   ' * (len(level) + 1)
                for f in sorted(files):
                    _, ext = os.path.splitext(f)
                    if item == 'latex' and ext in latex_ignore_ext:
                        continue
                    print(f"{indent}├── {f}")

if __name__ == "__main__":
    print_clean_tree()