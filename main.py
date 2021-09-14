from functions import *


def main():
    # ---------------------------Familiarity with data---------------------------
    dataframe = create_data()
    # print(tracks_per_class(dataframe))
    # dataframe = hist_tracks_per_class(dataframe)
    # ---------------------------draw data---------------------------------------
    # draw_first_track(dataframe)
    # draw_b(dataframe)
    # draw_c(dataframe)
    # draw_d(dataframe)
    # dataframe = add_energies(dataframe)
    #
    # show_confusion_matrix_f1_score(dataframe, [1, 16])
    # types_1_16 = select(dataframe, [1, 16])
    # random_forest(types_1_16, types_1_16['class'].values, [1, 16])
    # show_confusion_matrix_f1_score(dataframe, [3, 9])
    # types_3_9 = select(dataframe, [3, 9])
    # random_forest(types_3_9, types_3_9['class'].values, [3, 9])
    # show_confusion_matrix_f1_score(dataframe, [5, 6])
    # types_5_6 = select(dataframe, [5, 6])
    # random_forest(types_5_6, types_5_6['class'].values, [5, 6])
    # show_confusion_matrix_f1_score(dataframe, [12, 15])
    # types_12_15 = select(dataframe, [12, 15])
    # random_forest(types_12_15, types_12_15['class'].values, [12, 15])
    # --------------------------------------7----------------------------------
    # show_confusion_matrix_f1_score(dataframe, [1, 4, 7, 10])
    # show_confusion_matrix_f1_score_by_energy(dataframe, [1, 4, 7, 10])
    # types = select(dataframe, [1, 4, 7, 10])
    # random_forest(types, types['class'].values, [1, 4, 7, 10])
    #  ====3====
    compare(dataframe, [1, 4, 7, 10])




if __name__ == '__main__':
    main()
