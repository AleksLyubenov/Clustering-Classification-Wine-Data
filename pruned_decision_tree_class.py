from helper_functions import *
from decision_tree_class import *

class pruned_decision_tree(decision_tree):

    def __init__(self, tree, df_train, df_val, ml_task):
        self.tree = self.post_pruning(tree, df_train, df_val, ml_task)
        
        
    def filter_df(self, df, question):
        feature, comparison_operator, value = question.split()

        # continuous feature
        if comparison_operator == "<=":
            df_yes = df[df[feature] <= float(value)]
            df_no = df[df[feature] > float(value)]
        # categorical feature
        else:
            df_yes = df[df[feature].astype(str) == value]
            df_no  = df[df[feature].astype(str) != value]
        
        return df_yes, df_no
        
        
    def determine_leaf(self, df_train, ml_task):
        if ml_task == "regression":
            return df_train.target_label.mean()
        
        # classification
        else:
            return df_train.target_label.value_counts().index[0]
            
            
    def determine_errors(self, df_val, tree, ml_task):
        predictions = self.make_predictions(df_val, tree)
        actual_values = df_val.target_label
        
        if ml_task == "regression":
            # mean squared error
            return ((predictions - actual_values) **2).mean()
        else:
            # number of errors
            return sum(predictions != actual_values)
            
       
    def pruning_result(self, tree, df_train, df_val, ml_task):
        leaf = self.determine_leaf(df_train, ml_task)
        errors_leaf = self.determine_errors(df_val, leaf, ml_task)
        errors_decision_node = self.determine_errors(df_val, tree, ml_task)

        if errors_leaf <= errors_decision_node:
            return leaf
        else:
            return tree


    def post_pruning(self, tree, df_train, df_val, ml_task):
        question = list(tree.keys())[0]
        yes_answer, no_answer = tree[question]

        # base case
        if not isinstance(yes_answer, dict) and not isinstance(no_answer, dict):
            return self.pruning_result(tree, df_train, df_val, ml_task)
            
        # recursive part
        else:
            df_train_yes, df_train_no = self.filter_df(df_train, question)
            df_val_yes, df_val_no = self.filter_df(df_val, question)
            
            if isinstance(yes_answer, dict):
                yes_answer = self.post_pruning(yes_answer, df_train_yes, df_val_yes, ml_task)
                
            if isinstance(no_answer, dict):
                no_answer = self.post_pruning(no_answer, df_train_no, df_val_no, ml_task)
            
            tree = {question: [yes_answer, no_answer]}
        
            return self.pruning_result(tree, df_train, df_val, ml_task)     

    
