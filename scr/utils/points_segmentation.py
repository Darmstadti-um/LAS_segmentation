x_min_p = [] #списки с границами сегментов
x_max_p = []
y_min_p = []
y_max_p = []
z_min_p = []
z_max_p = []
seg_name = Day2_4_cloud1_seg #имя сегмента
base_df = Day2_4_cloud1 #исходный файл
tolerance = 1 #погрешность
points_df = base_df
for i in range (len(seg_name)):
  #segments = pd.concat([points_seg])

  xmin = (seg_name.iloc[i, seg_name.columns.get_loc('center_x')]) - (seg_name.iloc[i, seg_name.columns.get_loc('size_x')])/2

  xmax = (seg_name.iloc[i, seg_name.columns.get_loc('center_x')]) + (seg_name.iloc[i, seg_name.columns.get_loc('size_x')])/2

  ymin = (seg_name.iloc[i, seg_name.columns.get_loc('center_y')]) - (seg_name.iloc[i, seg_name.columns.get_loc('size_y')])/2

  ymax = (seg_name.iloc[i, seg_name.columns.get_loc('center_y')]) + (seg_name.iloc[i, seg_name.columns.get_loc('size_y')])/2

  zmin = (seg_name.iloc[i, seg_name.columns.get_loc('center_z')]) - (seg_name.iloc[i, seg_name.columns.get_loc('size_z')])/2

  zmax = (seg_name.iloc[i, seg_name.columns.get_loc('center_z')]) + (seg_name.iloc[i, seg_name.columns.get_loc('size_z')])/2

  x_min_p.append(xmin)
  x_max_p.append(xmax)
  y_min_p.append(ymin)
  y_max_p.append(ymax)
  z_min_p.append(zmin)
  z_max_p.append(zmax)

#for i in range(len(seg_name)):

  points_seg = points_df[(points_df['X'] >= x_min_p[i] - tolerance) & (points_df['X'] <= x_max_p[i] + tolerance) & (points_df['Y'] >= y_min_p[i] - tolerance) & (points_df['Y'] <=  y_max_p[i] + tolerance) & (points_df['Z'] >= z_min_p[i] - tolerance) & (points_df['Z'] <= z_max_p[i] + tolerance)]
  points_seg['center_x'] = seg_name.iloc[i, seg_name.columns.get_loc ('center_x')]
  points_seg['center_y'] = (seg_name.iloc[i, seg_name.columns.get_loc('center_y')])
  points_seg['center_z'] = (seg_name.iloc[i, seg_name.columns.get_loc('center_z')])
  points_seg['size_x'] = (seg_name.iloc[i, seg_name.columns.get_loc('size_x')])
  points_seg['size_y'] = (seg_name.iloc[i, seg_name.columns.get_loc('size_y')])
  points_seg['size_z'] = (seg_name.iloc[i, seg_name.columns.get_loc('size_z')])
  points_seg['yaw'] = (seg_name.iloc[i, seg_name.columns.get_loc('yaw')])
  points_seg['class'] = (seg_name.iloc[i, seg_name.columns.get_loc('class')])

  date = pd.concat([date, points_seg], ignore_index=1)
