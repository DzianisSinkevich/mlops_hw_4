from catboost.datasets import titanic
import pandas as pd
import os


def save_file(df, dir_path, file_name):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    try:
        file_path = os.path.join(dir_path, file_name)
        df.to_csv(file_path, index=False)
        print("File " + file_path + " created successfully.")
    except IOError:
        print("Error uccured while creating file " + file_path + " .")


# загрузка данных
train, test = titanic()

save_file(train, 'datasets', 'titanic_train.scv')
save_file(test, 'datasets', 'titanic_test.scv')

# обработка данных
# # заполним данные о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
# train['Sex'] = train['Sex'].apply(lambda x: 0 if 'male' else 1)
# test['Sex'] = test['Sex'].apply(lambda x: 0 if 'male' else 1)
# # в признаке "Возраст" много пропущенных (NaN) значений, заполним их средним значением возраста
# train['Age'] = train['Age'].fillna(train.Age.mean())
# test['Age'] = test['Age'].fillna(train.Age.mean())

# # запишем созданные датасеты во внешние csv-файлы
# train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']].to_csv('data_train.csv', index=False)
# test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].to_csv('data_test.csv', index=False)
