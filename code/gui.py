#! /usr/bin/env python3
import sys
import tkinter as tk
from tkinter import scrolledtext as st
from tkinter import ttk
from tkinter import N,E,W,S,END

class gui:
    def __init__(self, root):
        self.root = root
        self.boxValue = tk.StringVar() 
        self.boxValue.trace("w", self._comboCallback)
        self._createComponents()

    def _createComponents(self):
        self.root.title('Emotion Classifier')
        self.root.resizable(False, False)

        self.tBox = st.ScrolledText(self.root)
        self.fQuery = tk.Entry(self.root)
        self.comboBox = ttk.Combobox(self.root, textvariable=self.boxValue, state='readonly')
        self.bClassify = tk.Button(self.root, text="Classify", command=self._clickCallback)
        self.lQuery = tk.Label(self.root, text="Query:")
        self.lClassifier = tk.Label(self.root, text="Classifier:")
        self.bExit = tk.Button(self.root, text="Quit", command = self._quitCallback)

        self.tBox.grid(row=0, column=0, columnspan=24)
        self.lClassifier.grid(row=1, column=0, sticky=W)
        self.comboBox.grid(row=1, column=1, columnspan=6, sticky=E)
        self.bExit.grid(row=1, column=23, columnspan=1, sticky=W+E)
        self.lQuery.grid(row=2, column=0, sticky=W)
        self.fQuery.grid(row=2, column=1, columnspan=22, sticky=W+E+N+S)
        self.bClassify.grid(row=2, column=23, columnspan=1, sticky=W+E)
        # self.status.grid(row=3, column=0, columnspan=12, sticky=W+E+N+S)
        # self.emotion.grid(row=3,column=12, columnspan=12, sticky=W+E+N+S)
        
        self.tBox.configure(state="disabled")
        self.comboBox["values"] = ('Naive Bayes', 'SVM', 'VSM')
        self.comboBox.current(0)

        self.fQuery.bind('<Return>',self._returnCallback)

    def _clickCallback(self):
        if self.fQuery.get():
            msg = self.fQuery.get()
            self.tBox.configure(state="normal")
            #add prediction code here
            self.tBox.insert(END, "Text: "+msg+"\n")
            self.tBox.configure(state="disabled")
            self.fQuery.delete(0,len(msg))

    def _quitCallback(self):
        print('exiting...')
        self.root.destroy()
        sys.exit(0)

    def _returnCallback(self, event):
        self._clickCallback()

    def _comboCallback(self, *args):
        #set new classifier here
        print(self.comboBox["values"].index(self.boxValue.get()))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    obj = gui(root)
    obj.run()
