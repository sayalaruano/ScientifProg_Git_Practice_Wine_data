import pandas as pd

def data_formatting(data):
    wine_data = pd.read_excel(data, header = None)
    class_1 = [1] * 59
    class_2 = [2] * 71
    class_3 = [3] * 48
    class_vector = class_1 + class_2 + class_3

    wine_data_with_class = wine_data.assign(Classes = class_vector)
    return wine_data_with_class

wine_data = data_formatting("Wine.xlsx")
#print(wine_data)





