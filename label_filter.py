

import pandas as pd


class Label_Filter():
    def __init__(self, csv_path):
        self.csv_path = csv_path

        #  read the csv file from the csv_path
        self.df = pd.read_csv(self.csv_path)
        # # CSV Headers: name,id_org,id_40,new_name,id_super
        # print(self.df.head())        
        pass
    
    # get original label name and id
    def get_label_name_orig(self, id_org):        
        return self.df.loc[self.df['id_org'] == id_org, 'name'].values[0]
    def get_label_id_orig(self, id_org):        
        return id_org

    # get 40 class label name and id
    def get_label_name_40(self, id_org):
        return self.df.loc[self.df['id_org'] == id_org, 'new_name'].values[0]    
    def get_label_id_40(self, id_org):
        return self.df.loc[self.df['id_org'] == id_org, 'id_40'].values[0]    

    