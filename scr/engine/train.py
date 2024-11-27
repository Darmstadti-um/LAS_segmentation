# train_data = https://drive.google.com/file/d/15sBqMJ3kWtRNj1fpo49UMoAl8mr2UL71/view?usp=drive_link
df = pd.read_csv('path')
df = df.drop("Unnamed: 0",axis = 'columns')
df = df.dropna()
x_train = df[["X", "Y", "Z"]]
y_train = df[["center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "yaw", "class"]]
df.loc[df['class'] == 'vegetation', 'class'] = 1
df.loc[df['class'] == 'LEP_metal', 'class'] = 2
df.loc[df['class'] == 'LEP_prom', 'class'] = 3

y_train = df[["center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "yaw", "class"]]
x_train = df[["X", "Y", "Z"]]

x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')

model.fit(x_train, y_train, epochs=N) #N устанавливается самостоятельно

model.save('model.h5')
