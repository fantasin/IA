import pandas as pd
import numpy as np

class DataManager:
    def __init__(self,path):
        if "csv" == path[-3:]:
            self.df = pd.read_csv(path)


    def get_columns_name(self):
        """Retourne la liste des noms des columns"""
        return list(self.df.columns)

    def remove_columns(self, liste_name_columns):
        self.df = self.df.drop(liste_name_columns, axis=1)

    def remplace_value(self,column_name,list_old_value,new_value):
        self.df[column_name] = self.df[column_name].replace(list_old_value,new_value)

    def remplace_nan(self,column_name,new_value):
        if type(self.df[column_name][0]) == pd._libs.interval.Interval:
            self.df[column_name] = self.df[column_name].astype('string')
            self.df[column_name] = self.df[column_name].fillna(new_value)
            self.df[column_name] = self.df[column_name].astype('category')
        else:
            self.df[column_name] = self.df[column_name].fillna(new_value)

    def one_hot_encoding(self,column_name,suffix = None):
        tmp = pd.get_dummies(self.df[column_name],suffix)
        self.df = self.df.drop([column_name], axis=1)
        for name in list(tmp.columns):
            self.df[name] = tmp[name]

    def apply_function_column(self, column_name, function):
        self.df[column_name] = self.df[column_name].apply(function)

    def create_intervall_bins(self,liste_tuple_index):
        """Crée un interval index pour bin avec une liste de tuple par exemple [(1, 8), (8, 16), (16, 24),(24,30),(30,100)] """
        return pd.IntervalIndex.from_tuples(liste_tuple_index)

    def divise_en_categorie(self,bins,name):
        self.df[name] = pd.cut(self.df[name], bins)

    def get_mode(self,column_name,dropna = True):
        return self.df[column_name].mode(dropna=dropna)


    def print(self):
        print(self.df)

    def print_all_unique(self):
        for name in list(self.df.columns):
            print(name + ':  ', end="")
            print(self.df[name].unique())

    def get_unique(self,column_name):
        return self.df[column_name].unique()

    def shuffle(self):
        self.df  = self.df.sample(frac=1)

    def get_dataFrame(self):
        return self.df

    def quick_analyse(self,liste_to_analyze = None):
        if liste_to_analyze == None:
            liste_to_analyze = self.get_columns_name()

        for column_name in liste_to_analyze:
            print("*"*20)
            print("="*3 + ">" + column_name)
            print(self.df[column_name].value_counts(dropna=False))
            print("----")




