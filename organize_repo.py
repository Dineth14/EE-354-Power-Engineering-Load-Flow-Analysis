import os
import shutil

def organize():
    print("Organizing repository...")
    
    # Define the desired structure
    # Key: Folder Name, Value: List of files to move into it
    structure = {
        "src": [
            "Gauss_seidal_method.py", 
            "Newton_Raphson_Method.py", 
            "Load_Flow_Comparison.py"
        ],
        "notebooks": [
            "Load_Flow_Analysis.ipynb"
        ],
        "examples": [
            # The simple example files are already created in examples/ 
            # by the previous steps, so we don't need to move them.
        ]
    }
    
    base_dir = os.getcwd()
    
    for folder, files in structure.items():
        target_dir = os.path.join(base_dir, folder)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Created directory: {folder}/")
            
        for filename in files:
            src_path = os.path.join(base_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                print(f"Moved {filename} -> {folder}/")

if __name__ == "__main__":
    organize()
    print("Done.")