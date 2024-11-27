#Для извлечения характеристик каждого сегмента каждого файла выполнить следующее:
raw_data = pd.read_csv('/content/train.csv')
data_backup = raw_data.copy()
Day2_1_cloud0_seg = raw_data.loc[raw_data['file_name']=='Day2_1_cloud0.las'] #имя файла и сегмента меняется, соответственно
