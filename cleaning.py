
def clean_data(DataFrame):
    dimensions = [dimension for dimension in DataFrame.columns]

    for dimension in dimensions:
        indexes = DataFrame[DataFrame[dimension] == '?'].index
        DataFrame.drop(indexes , inplace=True)
    return DataFrame

def delete_dimension(DataFrame, index):
    indexes = list(range(len(DataFrame.columns)))
    del indexes[index]
    DataFrame = DataFrame.iloc[:, indexes]
    return DataFrame
