#преобразует las файл в датафрейм с нормальной координатной сеткой в которой составлена разметка
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
