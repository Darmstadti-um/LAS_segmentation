def coord_norm(f_name, path):
   globals()[f_name] = laspy.read(path)
   offset = globals()[f_name].header.offsets
   result = globals()[f_name]
   globals()[f_name] = globals()[f_name].xyz
   globals()[f_name] = pd.DataFrame(globals()[f_name], columns = ['X', 'Y', 'Z'])
   globals()[f_name].X -= offset[0]
   globals()[f_name].Y -= offset[1]
   globals()[f_name].Z -= offset[2]
   return result

coord_norm(f_name, path) 

#ИЛИ 
'''
import os
import zipfile
import laspy
import pandas as pd

def coord_norm(path):
    las = laspy.read(path)
    offset = las.header.offsets
    df = pd.DataFrame(las.xyz, columns=['X', 'Y', 'Z'])
    df['X'] -= offset[0]
    df['Y'] -= offset[1]
    df['Z'] -= offset[2]
    return df

def process_zip(zip_path, output_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    return {os.path.splitext(os.path.basename(f))[0]: coord_norm(os.path.join(root, f))
            for root, _, files in os.walk(output_folder)
            for f in files if f.endswith('.las')}
            '''
x = np.asarray(df).astype('float32')
model.predict(x)
