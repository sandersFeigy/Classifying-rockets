import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def create_data():
    dataframe = pd.read_csv('train.csv')
    dataframe.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    dataframe.set_index('index', inplace=True)
    dataframe.drop('targetName', axis=1, inplace=True)
    return dataframe


def add_distance(dataframe):
    track_length = []
    for i in range(0, len(dataframe)):
        row = dataframe.loc[[i]]

        z_start = row.posZ_0[i]
        index_null = row.isnull()
        index_no = end_of_rocket_trajectory(dataframe, index_null, i, "index")
        x_end, y_end, z_end = dataframe.iloc[[i], [index_no - 6]].loc[i][0], dataframe.iloc[[i], [index_no - 5]].loc[i][
            0], \
                              dataframe.iloc[[i], [index_no - 4]].loc[i][0]
        track_length += [math.sqrt(x_end ** 2 + y_end ** 2 + (z_end - z_start) ** 2)]
    dataframe['distance'] = track_length
    return dataframe


def tracks_per_class(dataframe):
    return dataframe.groupby(dataframe['class']).agg('size')


def end_of_rocket_trajectory(df, index_null, i, col_name):
    index_null = index_null[index_null.any(axis=1)].idxmax(axis=1)
    if str(index_null).find(col_name) == -1:
        return len(df.columns) - 3
    else:
        return df.columns.get_loc(index_null[i])


def hist_tracks_per_class(dataframe):
    dataframe = add_distance(dataframe)
    hists = dataframe.groupby(dataframe["class"]).agg({'distance': list})
    for i in range(1, 25):
        plt.hist(list(hists.iloc[i]['distance']))
        plt.show()
    return dataframe


def draw_rockets_trajectory(types):
    colors = ["red", "pink", "black", "yellow", "green", "orange"]
    for i in range(len(types)):
        index_number = end_of_rocket_trajectory(types, types.iloc[[i]].isnull(), i, "index")
        plt.plot(np.array(types.iloc[[i], 1: index_number:7])[0], np.array(types.iloc[[i], 3: index_number:7])[0],
                 color=colors[types.loc[[i], ["class"]].loc[i][0] - 1])
    plt.show()


def draw_first_track(dataframe):
    index_no = end_of_rocket_trajectory(dataframe, dataframe.iloc[[0]].isnull(), 0, "index")
    plt.plot(np.array(dataframe.iloc[[0], 1: index_no:7])[0], np.array(dataframe.iloc[[0], 3: index_no:7])[0])
    plt.show()


def draw_b(dataframe):
    types_1_to_6 = dataframe[dataframe["class"].isin([1, 2, 3, 4, 5, 6])][:50].reset_index()
    types_1_to_6.drop('index', axis=1, inplace=True)
    draw_rockets_trajectory(types_1_to_6)


def draw_c(dataframe):
    types_1_6 = dataframe[dataframe["class"].isin([1, 6]) & dataframe["posX_29"].notnull()][:50].reset_index()
    types_1_6.drop('index', axis=1, inplace=True)
    draw_rockets_trajectory(types_1_6)


def draw_d(dataframe):
    dataframe["max_height"] = dataframe[dataframe.columns[3::7]].max(axis=1)
    types_1_6 = dataframe[dataframe["class"].isin([1, 6]) & dataframe["posX_29"].notnull() & (
            dataframe["posZ_0"] < dataframe["max_height"]) &
                          (dataframe["posZ_29"] < dataframe["max_height"])][
                :50].reset_index()
    types_1_6.drop('index', axis=1, inplace=True)
    draw_rockets_trajectory(types_1_6)


def classified_types(training, class_types):
    training = add_max_height(training)
    d = dict()
    for i in range(len(class_types)):
        mean_height = training[training["class"] == class_types[i]]["max_height"].mean(axis=0, skipna=True)
        name = "type_" + str(class_types[i])
        d.update({name: {'max_height': mean_height}})
    return d


def classified_types_energy(training, class_types):
    training = add_energy(training)
    d = dict()
    for i in range(len(class_types)):
        mean_energy = training[training["class"] == class_types[i]]['energy0'].mean(axis=0, skipna=True)
        name = "type_" + str(class_types[i])
        d.update({name: {'energy0': mean_energy}})
    return d


def add_max_height(df):
    df["max_height"] = df[df.columns[3::7]].max(axis=1)
    return df


def guess_type(dic, test_set, types):
    test_set = add_max_height(test_set)
    arr = []
    for i in range(len(test_set)):
        m, typ = math.inf, 0
        for j in types:
            close = abs(test_set.loc[i]['max_height'] - dic['type_' + str(j)]['max_height'])
            if close < m:
                m = close
                typ = j
        arr += [typ]
    return arr


def rull_based_energy(dic, test_set, types):
    test_set = add_energy(test_set)
    arr = []
    for i in range(len(test_set)):
        m, typ = math.inf, 0
        for j in types:
            close = abs(test_set.loc[i]['energy0'] - dic['type_' + str(j)]['energy0'])
            if close < m:
                m = close
                typ = j
        arr += [typ]
    return arr


def test(guess_class_set, class_set):
    bad, good = 0, 0
    for i in range(len(guess_class_set)):
        if guess_class_set[i] == class_set[i]:
            good += 1
        else:
            bad += 1
    return good / len(guess_class_set) * 100


# Function to calculate kinetic Energy
def kinetic_energy(v):
    return 0.5 * v ** 2


# Function to calculate Potential Energy
def potential_energy(h):
    return 9.8 * h


def energy(v, h):
    return kinetic_energy(v) + potential_energy(h)


def add_energies(df):
    new_df = pd.concat([df, pd.DataFrame(columns=["energy"])])
    for i in range(len(new_df)):
        if i % 1000 == 0:
            print(i)
        energies = []
        for j in range(30):
            v, h = math.sqrt(
                new_df.iloc[i]["velX_" + str(j)] ** 2 + new_df.iloc[i]["velY_" + str(j)] ** 2 + new_df.iloc[i][
                    "velZ_" + str(j)] ** 2), new_df.iloc[i]["posZ_" + str(j)]
            energ = energy(v, h)
            energies += [energ]
        new_df.at[i, "energy"] = energies
    return new_df


def add_energy(df):
    new_df = pd.concat([df, pd.DataFrame(columns=["energy" + str(j) for j in range(15)])])
    for i in range(len(new_df)):
        for j in range(30):
            v, h = math.sqrt(
                new_df.iloc[i]["velX_" + str(j)] ** 2 + new_df.iloc[i]["velY_" + str(j)] ** 2 + new_df.iloc[i][
                    "velZ_" + str(j)] ** 2), new_df.iloc[i]["posZ_" + str(j)]
            energ = energy(v, h)
            new_df.at[i, "energy" + str(j)] = energ
    return new_df


def random_forest(df, y, types):
    imp = SimpleImputer(strategy="most_frequent")
    X = imp.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('random forest', types)
    print('confusion matrix', confusion_matrix(y_test, y_pred))
    print('F1 score', f1_score(y_test, y_pred, labels=[len(types)],average='macro', zero_division=1))


def random_forest_train(df, y, types):
    imp = SimpleImputer(strategy="most_frequent")
    X = imp.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print('random forest', types)
    print('confusion matrix', confusion_matrix(y_train, y_pred))
    print('F1 score', f1_score(y_train, y_pred, labels=[len(types)],average='macro', zero_division=1))


def select(dataframe, types):
    dataframe = dataframe[dataframe["class"].isin(types)].reset_index()
    dataframe.drop('index', axis=1, inplace=True)
    return dataframe


def show_confusion_matrix_f1_score(dataframe, types):
    df_types = select(dataframe, types)
    twenty_percent = int(len(df_types) * (1 / 5))
    idx_twenty_percent = random.sample(range(len(df_types)), twenty_percent)
    idx_twenty_percent.sort()
    training_set = df_types[~df_types.index.isin(idx_twenty_percent)].reset_index()
    training_set.drop('index', axis=1, inplace=True)
    test_set = df_types[df_types.index.isin(idx_twenty_percent)].reset_index()
    test_set.drop('index', axis=1, inplace=True)
    class_test_set = test_set["class"]
    del test_set["class"]
    dic = classified_types(training_set, types)
    guess = guess_type(dic, test_set, types)
    success = test(guess, class_test_set)
    print('rule base', types)
    print('confusion matrix', confusion_matrix(class_test_set, guess))
    print('F1 score', f1_score(class_test_set, guess, labels=[len(types)],average='macro', zero_division=1))


def show_confusion_matrix_f1_score_by_energy(dataframe, types):
    df_types = select(dataframe, types)
    twenty_percent = int(len(df_types) * (1 / 5))
    idx_twenty_percent = random.sample(range(len(df_types)), twenty_percent)
    idx_twenty_percent.sort()
    training_set = df_types[~df_types.index.isin(idx_twenty_percent)].reset_index()
    training_set.drop('index', axis=1, inplace=True)
    test_set = df_types[df_types.index.isin(idx_twenty_percent)].reset_index()
    test_set.drop('index', axis=1, inplace=True)
    class_test_set = test_set["class"]
    del test_set["class"]
    dic = classified_types_energy(training_set, types)
    guess = rull_based_energy(dic, test_set, types)
    # success = test(guess, class_test_set)
    print('rule base', types)
    print('confusion matrix', confusion_matrix(class_test_set, guess))
    print('F1 score', f1_score(class_test_set, guess, labels=[len(types)],average='macro', zero_division=1))


def compare(dataframe, array_types):
    types = select(dataframe, array_types)
    print('test set:')
    random_forest(types, types['class'].values, array_types)
    print('------------compare-------------------')
    print('training set:')
    random_forest_train(types, types['class'].values, array_types)
