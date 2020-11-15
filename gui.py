import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from embedding import *
from gensim_word2vec import *
from word2vec import *
from helper_functions import *
from random_forest_class import *

embeddedText = TRAIN = TEST = simTRAIN = simTEST = embeddingAlgorithm = classifier = None

class UI_MainWindow(object):
    
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(800, 400)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.DescriptionTextBox = QtWidgets.QTextEdit(self.centralwidget)
        self.DescriptionTextBox.setGeometry(QtCore.QRect(80, 80, 640, 140))
        self.DescriptionTextBox.setObjectName("DescriptionTextBox")
        
        self.DescriptionLabel = QtWidgets.QLabel(self.centralwidget)
        self.DescriptionLabel.setGeometry(QtCore.QRect(80, 50, 160, 20))
        self.DescriptionLabel.setObjectName("DescriptionLabel")
        
        self.EmbeddingComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.EmbeddingComboBox.setGeometry(QtCore.QRect(280, 230, 120, 20))
        self.EmbeddingComboBox.setObjectName("EmbeddingComboBox")
        self.EmbeddingComboBox.addItem("")
        self.EmbeddingComboBox.addItem("")
        
        self.EmbeddingLabel = QtWidgets.QLabel(self.centralwidget)
        self.EmbeddingLabel.setGeometry(QtCore.QRect(80, 230, 180, 13))
       
        self.ClassificationComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.ClassificationComboBox.setGeometry(QtCore.QRect(280, 280, 120, 20))
        self.ClassificationComboBox.setObjectName("ClassificationComboBox")
        self.ClassificationComboBox.addItem("")
        self.ClassificationComboBox.addItem("")
        self.ClassificationComboBox.addItem("")
        
        self.ClassificationLabel = QtWidgets.QLabel(self.centralwidget)
        self.ClassificationLabel.setGeometry(QtCore.QRect(80, 280, 180, 13))
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EmbeddingLabel.sizePolicy().hasHeightForWidth())

        self.EmbeddingLabel.setSizePolicy(sizePolicy)
        self.EmbeddingLabel.setObjectName("EmbeddingLabel")
        
        self.ClassificationLabel.setSizePolicy(sizePolicy)
        self.ClassificationLabel.setObjectName("EmbeddingLabel")
        
        self.EmbeddingPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.EmbeddingPushButton.setGeometry(QtCore.QRect(600, 230, 120, 20))
        self.EmbeddingPushButton.setObjectName("EmbeddingPushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        # connect EmbeddingPushButton to function on_click_embedding
        self.EmbeddingPushButton.clicked.connect(self.on_click_embedding)
        
        self.ClassificationPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ClassificationPushButton.setGeometry(QtCore.QRect(600, 280, 120, 20))
        self.ClassificationPushButton.setObjectName("ClassificationPushButton")
        # connect ClassificationPushButton to function on_click_classification
        self.ClassificationPushButton.clicked.connect(self.on_click_classification)

        self.RecommendationPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.RecommendationPushButton.setGeometry(QtCore.QRect(600, 330, 120, 20))
        self.RecommendationPushButton.setObjectName("RecommendationPushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        # connect RecommendationPushButton to function on_click_recommendation
        self.RecommendationPushButton.clicked.connect(self.on_click_recommendation)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")       
       
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.DescriptionLabel.setText(_translate("MainWindow", "Enter Wine Description Here:"))
        
        self.EmbeddingComboBox.setItemText(0, _translate("MainWindow", "Gensim Word2Vec"))
        self.EmbeddingComboBox.setItemText(1, _translate("MainWindow", "Naive Word2Vec"))
        self.EmbeddingLabel.setText(_translate("MainWindow", "Choose Text Embedding Algorithm:"))
        
        self.ClassificationComboBox.setItemText(0, _translate("MainWindow", "Decision Tree"))
        self.ClassificationComboBox.setItemText(1, _translate("MainWindow", "Pruned Decision Tree"))
        self.ClassificationComboBox.setItemText(2, _translate("MainWindow", "Random Forest"))
        self.ClassificationLabel.setText(_translate("MainWindow", "Choose Classifier:"))
        
        self.ClassificationPushButton.setText(_translate("MainWindow", "Get Classification"))
        self.EmbeddingPushButton.setText(_translate("MainWindow", "Embed Text"))
        self.RecommendationPushButton.setText(_translate("MainWindow", "Get Recommendation"))
       
    def on_click_embedding(self):   
        global embeddedText, TRAIN, TEST, simTRAIN, simTEST, embeddingAlgorithm, classifier
        
        msgBox = QtWidgets.QMessageBox()
        textboxValue = self.DescriptionTextBox.toPlainText()
        
        #Embed the text using the algorithm chosen by the user
        if textboxValue == '':
            msgBox.setText('You have not entered a wine description. Please type a description in the text box before proceeding.')
            msgBox.setWindowTitle('EmbeddingError')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
            returnValue = msgBox.exec()
        else:
            df = process_all(textboxValue)
            embeddingAlgorithm = str(self.EmbeddingComboBox.currentText())
            if embeddingAlgorithm == 'Gensim Word2Vec':
                TRAIN, TEST, simTRAIN, simTEST = gensim_w2v_embedding(df)
            else:
                #TRAIN, TEST, simTRAIN, simTEST  = naive_w2v_embedding(df)
                TRAIN = load_obj('TRAIN')
                TEST = load_obj('TEST')
                simTRAIN = load_obj('simTRAIN')
                simTEST = load_obj('simTEST')
            print('TEST VECTOR:\n{}\n'.format(TEST))    
            
            embeddedText = TEST.loc[TEST['target_label']=='X'].iloc[0, 0:-1]
            print(TRAIN, '\n\n', TEST)
            msgBox.setText('Your Wine Description:\n\n' + textboxValue + '\n\nText Successfully Embedded!')
            msgBox.setWindowTitle('WineDescription')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            
            returnValue = msgBox.exec()
            
            if returnValue == QtWidgets.QMessageBox.Ok:
                print('OK clicked')
                #self.DescriptionTextBox.setText("")
            else:
                print('Cancel clicked')
            
            
    def on_click_classification(self):     
        global embeddedText, TRAIN, TEST, simTRAIN, simTEST, embeddingAlgorithm, classifier
    
        msgBox = QtWidgets.QMessageBox()
        global embeddedText, embeddingAlgorithm, classifier, TRAIN, TEST
        
        if embeddedText is None:
            msgBox.setText('Please embed your text before proceeding.')
            msgBox.setWindowTitle('ClassificationError')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        else:
            classifier = str(self.ClassificationComboBox.currentText())
            if classifier == 'Decision Tree':
                if embeddingAlgorithm == 'Gensim Word2Vec':
                    clf = load_obj('DECISION_TREE_FULL_TRAIN_GENSIM')
                else:
                    clf = load_obj('DECISION_TREE_FULL_TRAIN_NAIVE')        
                prediction=clf.predict_example(TEST, clf.tree)
            
            elif classifier == 'Pruned Decision Tree':
                if embeddingAlgorithm == 'Gensim Word2Vec':
                    clf = load_obj('PRUNED_DECISION_TREE_FULL_TRAIN_GENSIM')
                else:
                    clf = load_obj('PRUNED_DECISION_TREE_FULL_TRAIN_NAIVE') 
                prediction=clf.predict_example(TEST, clf.tree)
                
            else:
                if embeddingAlgorithm == 'Gensim Word2Vec':
                    clf = load_obj('RANDOM_FOREST_FULL_TRAIN_GENSIM')
                else:
                    clf = load_obj('RANDOM_FOREST_FULL_TRAIN_NAIVE')     
                prediction = clf.random_forest_predict(TEST)
                prediction = prediction.values[0]

            print('Used {} with {} embeddings.'.format(classifier,embeddingAlgorithm))
            if prediction == 0.0:
                msgBox.setText('The wine you are describing is: ' + 'WHITE')
            else:
                msgBox.setText('The wine you are describing is: ' + 'RED')                
            msgBox.setWindowTitle('Classification')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        returnValue = msgBox.exec()
        
        if returnValue == QtWidgets.QMessageBox.Ok:
            print('OK clicked')
            #self.DescriptionTextBox.setText("")
        else:
            print('Cancel clicked')
            
    def on_click_recommendation(self):
        global embeddedText, TRAIN, TEST, simTRAIN, simTEST, embeddingAlgorithm, classifier

        msgBox = QtWidgets.QMessageBox()
        
        if embeddedText is None:
            msgBox.setText('Please embed your text before proceeding.')
            msgBox.setWindowTitle('RecommendationError')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        else:
            similar_wines = similar_descriptions(simTRAIN, simTEST, 5)
            
            str_main = ""
            for i in range(len(similar_wines)):
                str = '\nWine No. {} - {}'.format(i+1, similar_wines[i][0])
                str_main += str
            msgBox.setText(str_main)
                
            msgBox.setWindowTitle('Recommendations')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        
        returnValue = msgBox.exec()
        
        if returnValue == QtWidgets.QMessageBox.Ok:
            print('OK clicked')
            #self.DescriptionTextBox.setText("")
        else:
            print('Cancel clicked')
            
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    ui = UI_MainWindow()
    ui.setupUI(MainWindow)
    
    MainWindow.show()
    sys.exit(app.exec_())
