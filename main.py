from catboost.datasets import titanic
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder  # Импортируем One-Hot Encoding от scikit-learn


def save_file(df, dir_path, file_name):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    try:
        file_path = os.path.join(dir_path, file_name)
        df.to_csv(file_path, index=False)
        print("File " + file_path + " created successfully.")
    except IOError:
        print("Error uccured while creating file " + file_path + " .")


def read_file(file_path):
    try:
        df = pd.read_csv(file_path)

        print("File " + file_path + " readed successfully.")
        return df
    except IOError:
        print("Error uccured while readed file '" + file_path + "'.")


# загрузка данных
def read_dfs():
    train = read_file('datasets/titanic_train.scv')
    test = read_file('datasets/titanic_test.scv')
    return train, test


def save_dfs(train, test):
    save_file(train, 'datasets', 'titanic_train.scv')
    save_file(test, 'datasets', 'titanic_test.scv')


# замена NaN в Age на среднее значение возраста
def anti_none(train, test):
    train['Age'] = train['Age'].fillna(train.Age.mean())
    test['Age'] = test['Age'].fillna(train.Age.mean())
    return train, test


# кодирование данных о поле пассажира числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
def sex_convert(train, test):
    train['Sex'] = train['Sex'].replace(['male', 'female'], [0, 1])
    test['Sex'] = test['Sex'].replace(['male', 'female'], [0, 1])
    return train, test


def one_hot(df, column_names):
    # Создание Объекта OneHotEncoder() и его "обучение" .fit
    ohe = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False)
    ohe.fit(df[column_names])

    # Применяем трансформацию .transform и сохраняем результат в Dataframe
    ohe_feat = ohe.transform(df[column_names])
    df_ohe = pd.DataFrame(ohe_feat, columns=ohe.get_feature_names_out()).astype(int)
    print(df_ohe)
    return df_ohe


def main():
    # train, test = titanic()
    train, test = read_dfs()

    # 1
    # train, test = anti_none(train, test)

    # 2
    # train, test = sex_convert(train, test)

    # 3
    train = pd.concat([train, one_hot(train, ['Sex'])], axis=1).reindex(train.index)
    test = pd.concat([test, one_hot(test, ['Sex'])], axis=1).reindex(test.index)

    save_dfs(train, test)


main()
