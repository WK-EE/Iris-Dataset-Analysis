"""
* @author: Wael Khalil
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def read_dataset(filename):
    '''
    This function reads in the dataset of the filename that is passed in.
    :param filename: A string representing the filename of the dataset we are
    working with.
    :return: A dataframe representing the dataset.
    '''

    df = pd.read_csv(filename)
    return df


def visualize_two_features(our_dataset):
    '''
    This function displays two features at a time along with the class they
    belong to. It also plots the two features passed to it.
    :param our_dataset: A dataframe of the dataset we are working with.
    :return: This function does not return a value.
    '''
    plot = sns.pairplot(our_dataset, hue='species', palette='husl', height = 1.8)
    plot.fig.suptitle('Features Visualization', y=1)
    plt.show()

def two_way_merge_sort(arr: list, species_list: list, comparisons: list,
                       swaps: list):
    '''
    This function performs a two-way merge sort on the data passed to it.
    :param arr: An array (list) representing the data processed from file
    :param species_list: A list that would contain the transformed species
    column.
    :param comparisons: A list that would store the number of comparisons
    :param swaps: A list that would store the number of exchanges
    :return: This function does not have a return. It modifies the array
    passed to it as one of its arguments (which is a copy).
    '''

    if len(arr) > 1:

        # Finding the mid of the array
        mid = len(arr) // 2

        # Dividing the array elements into two sublists, left and right
        L = arr[:mid]
        R = arr[mid:]

        L_species = species_list[:mid]
        R_species = species_list[mid:]

        # Sorting the first half
        two_way_merge_sort(L, L_species, comparisons, swaps)

        # Sorting the second half
        two_way_merge_sort(R, R_species, comparisons, swaps)

        left_index = right_index = sorted_index = 0

        # Copy data to temp arrays L[] and R[]
        while left_index < len(L) and right_index < len(R):
            comparisons[0] += 1
            swaps[0] += 1
            if L[left_index] < R[right_index]:
                arr[sorted_index] = L[left_index]
                # this takes care of sorting the species column
                species_list[sorted_index] = L_species[left_index]
                left_index += 1

            else:
                arr[sorted_index] = R[right_index]
                # this takes care of sorting the species column
                species_list[sorted_index] = R_species[right_index]
                right_index += 1

            sorted_index += 1

        # Checking if any element was left
        while left_index < len(L):
            swaps[0] += 1
            arr[sorted_index] = L[left_index]
            # this takes care of sorting the species column
            species_list[sorted_index] = L_species[left_index]
            left_index += 1
            sorted_index += 1

        while right_index < len(R):
            swaps[0] += 1
            arr[sorted_index] = R[right_index]
            # this takes care of sorting the species column
            species_list[sorted_index] = R_species[right_index]
            right_index += 1
            sorted_index += 1


def sort_feature(dataset, column_name):
    '''
    This is the function that makes sure that the data passed in is transformed
    into a Python list object before it calls the sorting algorithm on our data.
    :param dataset: A dataframe representing the dataset we are working with
    :param column_name: A string representing the column name of the feature
    we will perform the sort on
    :return: A dataframe that will contain two columns, the sorted feature
    column, and the species class column.
    '''

    feature_data = dataset[column_name]
    feature_array = np.array(feature_data).tolist()
    species_array = np.array(dataset["species"]).tolist()

    # allows us to count comparisons and swaps
    comparisons = [0]
    swaps = [0]

    # calling our mergesort function
    two_way_merge_sort(feature_array, species_array, comparisons, swaps)

    # creating a dataframe out of our sorted data
    data = {column_name: feature_array, "species": species_array}
    sorted_feature = pd.DataFrame(data)

    # checking if directory exists
    # if not, then create one to output sorted data to it
    if os.path.isdir('Sorted_Data/'):
        pass
    else:
        os.mkdir('Sorted_Data/')

    # outputting sorted data to excel
    excel_filepath = f'Sorted_Data/sorted_{column_name}.xlsx'
    writer_obj = pd.ExcelWriter(excel_filepath, engine='xlsxwriter')
    sorted_feature.to_excel(writer_obj)
    writer_obj.save()

    return sorted_feature


def outlier_removal(dataset_class):
    '''
    This function uses the Z-score to detect and remove the outliers in the
    feature passed into the function
    :param dataset_class: A dataframee representing the class we are working
    with.
    :return: A dataframe that contains the class columns with outliers
    removed.
    '''

    # applying Z-score on our class to find our outliers
    z = np.abs(stats.zscore(dataset_class))

    # list to append the locations of our outliers
    l = []
    l.append(np.where(z >= 2.56))
    outlier_removed_row_index = l[0][0]
    outlier_removed_column_index = l[0][1]

    # the filtered outliers are found using a threshold of 2.5
    filtered_outliers = (z < 2.56).all(axis=1)
    removed_outliers_class_df = dataset_class[filtered_outliers]

    return removed_outliers_class_df, outlier_removed_row_index, outlier_removed_column_index

def plot_class_outliers(class_df, label ,outliers_arr):
    '''
    This function outputs a pairplot of the outliers in each feature
    of a specific class
    :param class_df: A dataframe representing the class we are working with
    :param label: A string representing the label column in our dataset.
    In our example the label column is 'species'
    :param outliers_arr: A python list that contains the outliers of that class
    :return: This function does not have a return value.
    '''

    # grabbing the class name to use it for our plot
    if 'setosa' in class_df.species.values:
        class_name = 'Setosa'

    elif 'versicolor' in class_df.species.values:
        class_name = 'Versicolor'

    else:
        class_name = 'Virginica'

    # dropping our class label column
    data = class_df.copy()
    data = data.drop(label, axis = 1)

    # creates a True or False column called outlier
    # the column cell will have a True if the row has any outliers and false
    # if none are found
    data['outlier'] = np.where(data['sepal_length'].isin(outliers_arr)| data['sepal_width'].isin(outliers_arr) |
             data['petal_length'].isin(outliers_arr) |
             data['petal_width'].isin(outliers_arr), True, False)

    plot = sns.pairplot(data, hue = 'outlier', kind='scatter', height = 1.9)
    plot.fig.suptitle(f'{class_name} Outlier Visualization', y = 1)
    plt.show()


def feature_ranking_chi_squared(dataset):
    '''
    This function performs the Chi-squared test on our dataset allowing to
    rank the features in it.
    :param dataset: A dataframe representing the dataset we are working with
    :return: A python list that contains the scores of our features
    '''

    array = dataset.values

    # splitting the features and the class label
    X = array[:, :4]
    Y = array[:, 4]

    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)

    np.set_printoptions(precision=3)
    features = fit.transform(X)

    # Summarize selected features
    #print(features[0:5, :])

    return fit.scores_

def fisher_discriminant(dataset,feature):
    '''
    This function calculates the Fisher score for the feature passed in
    :param dataset: A dataframe representing the dataset we are working with
    :param feature: A string representing the column name of our feature
    :return: A float representing the Fisher score of the feature
    '''

    setosa_class = dataset.loc[:49, feature]
    versicolor_class = dataset.loc[50:99, feature]
    virginica_class = dataset.loc[100:149, feature]
    FDR = ((setosa_class.mean() - versicolor_class.mean())**2) / ((setosa_class.std())**2 + (versicolor_class.std())**2)
    FDR = FDR + ((versicolor_class.mean() - virginica_class.mean())**2) / ((versicolor_class.std())**2 + (virginica_class.std())**2)
    FDR = FDR + ((virginica_class.mean() - setosa_class.mean())**2) / (
                (virginica_class.std())**2 + (setosa_class.std())**2)

    return FDR

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # the directory of our dataset csv file
    filename = "iris.csv"

    # reading our iris dataset
    our_dataset = read_dataset(filename)

    # plot that allows us visualize two features
    visualize_two_features(our_dataset)

    # sorting features, one feature at a time along with the species column
    # to allow us to interpret what we are seeing
    sorted_sepal_length_feature = sort_feature(our_dataset, "sepal_length")
    sorted_sepal_width_feature = sort_feature(our_dataset, "sepal_width")
    sorted_petal_length_feature = sort_feature(our_dataset, "petal_length")
    sorted_petal_width_feature = sort_feature(our_dataset, "petal_width")

    # printing sorted features
    print('Sorted Sepal Length\n',sorted_sepal_length_feature, '\n')
    print('Sorted Sepal Width\n',sorted_sepal_width_feature, '\n')
    print('Sorted Petal Length\n',sorted_petal_length_feature,'\n')
    print('Sorted Petal Width\n',sorted_petal_width_feature, '\n')


    # detecting outliers in setosa class
    setosa_class = our_dataset.iloc[:50, :4]
    setosa_outliers_removed, outliers_row_index, outliers_column_index = outlier_removal(setosa_class)
    setosa_outliers_values = []
    for i in range(len(outliers_row_index)):
        setosa_outliers_values.append(our_dataset.iloc[outliers_row_index[i], outliers_column_index[i]])
    print(f"The outliers of the Setosa class are {setosa_outliers_values}\n")
    plot_class_outliers(our_dataset.iloc[:50, :5], 'species', setosa_outliers_values)

    # detecting outliers in versicolor class
    versicolor_class = our_dataset.iloc[50:100, :4]
    versicolor_outliers_removed, outliers_row_index, outliers_column_index = outlier_removal(versicolor_class)
    versicolor_outliers_values = []
    for i in range(len(outliers_row_index)):
        versicolor_outliers_values.append(versicolor_class.iloc[outliers_row_index[i], outliers_column_index[i]])
    print(f"The outliers of the Versicolor class are {versicolor_outliers_values}\n")
    plot_class_outliers(our_dataset.iloc[50:100, :5], 'species', versicolor_outliers_values)

    # detecting outliers in virginica class
    virginica_class = our_dataset.iloc[100:150, :4]
    virginica_outliers_removed, outliers_row_index, outliers_column_index = outlier_removal(virginica_class)
    virginica_outliers_values = []
    for i in range(len(outliers_row_index)):
        virginica_outliers_values.append(virginica_class.iloc[outliers_row_index[i], outliers_column_index[i]])
    print(f"The outliers of the Virginica class are {virginica_outliers_values}\n")
    plot_class_outliers(our_dataset.iloc[100:150, :5], 'species', virginica_outliers_values)

    # ranking our features using chi-squared
    features_arr = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    features_ranking_scores = feature_ranking_chi_squared(our_dataset)
    data = {'Feature':features_arr, 'Chi-Squared Ranking Score':features_ranking_scores}
    ranking_df = pd.DataFrame(data)
    print(ranking_df)

    # ranking our features using FDR
    sepal_length_fdr = fisher_discriminant(our_dataset, 'sepal_length')
    sepal_width_fdr = fisher_discriminant(our_dataset, 'sepal_width')
    petal_length_fdr = fisher_discriminant(our_dataset, 'petal_length')
    petal_width_fdr = fisher_discriminant(our_dataset, 'petal_width')

    # forming a dataframe that demonstrates the rankings of our features
    features_ranking_fdr_arr = []
    features_ranking_fdr_arr.append(sepal_length_fdr)
    features_ranking_fdr_arr.append(sepal_width_fdr)
    features_ranking_fdr_arr.append(petal_length_fdr)
    features_ranking_fdr_arr.append(petal_width_fdr)
    data = {'Feature':features_arr, 'FDR Ranking Score':features_ranking_fdr_arr}
    fdr_ranking_df = pd.DataFrame(data)
    print('\n',fdr_ranking_df,'\n')
