#! /usr/bin/env python3
import sys
import tkinter as tk
from tkinter import scrolledtext as st
from tkinter import ttk
from tkinter import N,E,W,S,END
from vsm import vsm
from nb import nb
from svm import svm
import random

class gui:
    def __init__(self, root):
        self.root = root
        self.boxValue = tk.StringVar() 
        self.boxValue.trace("w", self._comboCallback)
        self.emoVar = tk.StringVar()
        self.emoVar.set("Emotion?")

        self.classVar = tk.StringVar()
        self.classVar.set("VSM")
        self.img = tk.PhotoImage()
        self.img.configure(file='../images/logo.png')

        self._createComponents()        
        self.trainClassifiers()
        

    def trainClassifiers(self):
        """
        Function pre creates objects for all classifiers, so that prediction is fast.
        An instance dict of those models is created so that they can be indexed easily.
        arguments: none
        return: none
        """
        
        self.which = {0:vsm(),1:nb(),2:svm()}
        for i in self.which.keys():
            self.which[i].fit()


    def _createComponents(self):
        """
        Create and initialize all gui components.
        arguments: none
        return: none
        """
        self.root.title('Emotion Classifier')
        self.root.resizable(False, False)

        self.tBox = st.ScrolledText(self.root,font=("Ubuntu",12))
        self.fQuery = tk.Entry(self.root)
        self.comboBox = ttk.Combobox(self.root, textvariable=self.boxValue, state='readonly')
        self.bClassify = tk.Button(self.root, text="Classify", command=self._clickCallback)
        self.lQuery = tk.Label(self.root, text="Query:")
        self.lClassifier = tk.Label(self.root, text="Classifier:")
        self.bExit = tk.Button(self.root, text="Quit", command = self._quitCallback)
        self.emo = tk.Label(self.root, font=("Ubuntu", 24), textvariable=self.emoVar, fg="blue")
        self.whichClassifier = tk.Label(self.root, font=("Ubuntu", 18), textvariable=self.classVar, fg="blue")
        self.smiley = tk.Label(self.root,image=self.img)

        self.tBox.grid(row=0, column=0, columnspan=24)
        self.lClassifier.grid(row=1, column=0, sticky=W)
        self.comboBox.grid(row=1, column=1, columnspan=6, sticky=E)
        self.bExit.grid(row=1, column=23, columnspan=1, sticky=W+E)
        self.lQuery.grid(row=2, column=0, sticky=W)
        self.fQuery.grid(row=2, column=1, columnspan=22, sticky=W+E+N+S)
        self.bClassify.grid(row=2, column=23, columnspan=1, sticky=W+E)
        self.emo.grid(row=3,column=0,columnspan=6,sticky=W)
        self.smiley.grid(row=3,column=9,sticky=N+E+W+S)
        self.whichClassifier.grid(row=3,column=20,sticky=E)
        # self.status.grid(row=3, column=0, columnspan=12, sticky=W+E+N+S)
        # self.emotion.grid(row=3,column=12, columnspan=12, sticky=W+E+N+S)
        
        self.tBox.configure(state="disabled")
        self.comboBox["values"] = ('VSM', 'Naive Bayes', 'SVM')
        self.comboBox.current(0)

        self.fQuery.bind('<Return>',self._returnCallback)

        #create some tags
        self.tBox.tag_configure("anger", foreground="red")
        self.tBox.tag_configure("joy", foreground="green")
        self.tBox.tag_configure("sadness", foreground="gray")

    def _clickCallback(self):
        """
        Callback function to update gui when the a query has been entered
        arguments: none
        return: none
        """
        if self.fQuery.get():
            msg = self.fQuery.get()
            classifier = self.comboBox["values"].index(self.boxValue.get())
            emo = self.which[classifier].predict(msg)
            msg = msg+' [['+emo+']]   [['+self.boxValue.get()+']]'            
            self.emoVar.set(emo)
            self.img.configure(file="../images/"+emo+".png")
            self.tBox.configure(state="normal")
            self.tBox.insert(END, "Text: "+msg+"\n",emo)
            self.tBox.configure(state="disabled")
            self.fQuery.delete(0,len(msg))

    def _quitCallback(self):
        """
        Callback to quit application
        arguments: none
        return: none
        """
        print('exiting...')
        self.root.destroy()
        sys.exit(0)

    def _returnCallback(self, event):
        """
        Callback to handle <enter> action on textfield
        arguments: triggering event
        return: none
        """
        self._clickCallback()

    def _comboCallback(self, *args):
        """
        Callback to change classifier label, when new one selected from the combobox
        arguments: list of values in combobox
        return: none
        """
        #set new classifier here
        self.classVar.set(self.boxValue.get())
        print(self.comboBox["values"].index(self.boxValue.get()))

    def run(self):
        """
        Function to start main loop of application
        argument: none
        return: none
        """
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    obj = gui(root)
    obj.run()
