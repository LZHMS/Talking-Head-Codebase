from base import DataManager

class CodeTalkerDataManager(DataManager):
  
    def __init__(self,
                assistant,
                dataset_wrapper=None):
        super().__init__(assistant, dataset_wrapper)