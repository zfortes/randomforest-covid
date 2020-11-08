import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# pd.set_option('display.expand_frame_repr', False)
# X.head(1)
# print(X)

def ler_arquivo():
    return pd.read_excel('dataset.xlsx')


def preparar_dataset(df):
    df['SARS-Cov-2 exam result'] = [0 if a == 'negative' else 1 for a in df['SARS-Cov-2 exam result'].values]

    Y = df['SARS-Cov-2 exam result']

    df = df.drop([
        "SARS-Cov-2 exam result",
        "Patient ID",
        'Patient addmited to regular ward (1=yes, 0=no)',
        'Patient addmited to semi-intensive unit (1=yes, 0=no)',
        'Patient addmited to intensive care unit (1=yes, 0=no)',
        'Patient age quantile'
    ], axis=1)
    df = df.fillna(-10, axis=1)
    columns = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
    X = pd.get_dummies(df, prefix=columns, columns=columns)
    return X, Y


def preparar_dataset2(df):
    df['SARS-Cov-2 exam result'] = [0 if a == 'negative' else 1 for a in df['SARS-Cov-2 exam result'].values]

    Y = df['Patient addmited to regular ward (1=yes, 0=no)']

    df = df.drop([

        "Patient ID",
        'Patient addmited to regular ward (1=yes, 0=no)',
        'Patient addmited to semi-intensive unit (1=yes, 0=no)',
        'Patient addmited to intensive care unit (1=yes, 0=no)',
        'Patient age quantile'
    ], axis=1)
    df = df.fillna(-10, axis=1)
    columns = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
    X = pd.get_dummies(df, prefix=columns, columns=columns)
    return X, Y


def treina_classificador_random_forest(x_train, y_train):
    clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=40)
    clf.fit(x_train, y_train)
    return clf


def verifica_categorias_importantes(clf, x_train):
    columns = pd.DataFrame(clf.feature_importances_, index=x_train.columns,
                           columns=['importance']).sort_values('importance', ascending=False)
    pf = columns.head(10)

    feat_importances = pd.Series(clf.feature_importances_, index=x_train.columns)
    # feat_importances.nlargest(10).plot(kind='barh')
    # plt.title("Top 10 important features")
    # plt.show()
    print(pf)


def run():
    df = ler_arquivo()
    df2 = df.copy()
    x, y = preparar_dataset(df)
    print(x)
    print(len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=55)
    clf = treina_classificador_random_forest(x_train, y_train)
    pred_y = clf.predict(x_test)
    print(pred_y)
    pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print(f'Mean accuracy score: {accuracy:.3}')
    clf.score(x_test, y_test)
    print(classification_report(y_test, pred_y))
    verifica_categorias_importantes(clf, x_train)

    print("======================================================================================")
    print("                                             DF2                                      ")

    x, y = preparar_dataset2(df2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=55)
    clf = treina_classificador_random_forest(x_train, y_train)
    pred_y = clf.predict(x_test)
    print(pred_y)
    pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print(f'Mean accuracy score: {accuracy:.3}')
    clf.score(x_test, y_test)
    print(classification_report(y_test, pred_y))
    # verifica_categorias_importantes(clf, x_train)
    verifica_categorias_importantes(clf, x_train)
