import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Função responsável por ler o arquivo xlsx e salva-lo em uma variável como uma matriz
def ler_arquivo():
    return pd.read_excel('dataset.xlsx')


# Função resposável por preparar o dataset para que possa ser divido entre os dados para treino e os dados para teste
# Aqui colunas que não são importantes para o resultado foram removidas e as colunas que possuiam valores que não
# eram números foram convertidas para números.
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
    df = df.fillna(0, axis=1)
    columns = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
    X = pd.get_dummies(df, prefix=columns, columns=columns)
    return X, Y

# Possui o mesmo objetivo que a função preparar_dataset, porem esse é utilizado para a questão 2 onde se faz necessário
# alterar o valor de y para o treinamento.
def preparar_dataset2(df2, s):
    df = df2.copy()
    df['SARS-Cov-2 exam result'] = [0 if a == 'negative' else 1 for a in df['SARS-Cov-2 exam result'].values]

    Y = df[s]

    df = df.drop([
        "Patient ID",
        'Patient addmited to regular ward (1=yes, 0=no)',
        'Patient addmited to semi-intensive unit (1=yes, 0=no)',
        'Patient addmited to intensive care unit (1=yes, 0=no)',
        'Patient age quantile'
    ], axis=1)
    df = df.fillna(0, axis=1)
    columns = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
    X = pd.get_dummies(df, prefix=columns, columns=columns)
    return X, Y


# Função responsável por criar a instância do classificador Random Forest e em seguida treinar o algoritmo com o
# dataset preparado nos passoa anteriores
def treina_classificador_random_forest(x_train, y_train):
    clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=40)
    clf.fit(x_train, y_train)
    return clf

# Verica quais colunas (variáveis) apresentam a maior importância na tabela para a obtenção dos resultados.
def verifica_categorias_importantes(clf, x_train):
    columns = pd.DataFrame(clf.feature_importances_, index=x_train.columns,
                           columns=['importance']).sort_values('importance', ascending=False)
    pf = columns.head(10)
    print(pf)

# A função run executa os seguintes passos:
# 1 - Lê o arquivo
# 2 - Prepara o dataset
# 3 - Separa os dados em treino e teste
# 4 - Treina o classificador
# 5 - Faz uma predição e mede a acurácia do algoritmo.
# Esses passos são repetidos para todos os casos das questões.
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
    print("          Patient addmited to regular ward (1=yes, 0=no)                              ")

    x, y = preparar_dataset2(df2, 'Patient addmited to regular ward (1=yes, 0=no)')
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
    print("                 Patient addmited to semi-intensive unit (1=yes, 0=no)               ")

    x, y = preparar_dataset2(df2, 'Patient addmited to semi-intensive unit (1=yes, 0=no)')
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
    print("              Patient addmited to intensive care unit (1=yes, 0=no)              ")

    x, y = preparar_dataset2(df2, 'Patient addmited to intensive care unit (1=yes, 0=no)')
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
