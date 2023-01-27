import re
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import json
from nltk import word_tokenize
import math
from nltk.stem import WordNetLemmatizer
from numpy import dot
from numpy.linalg import norm
import pathlib


lemmatizer = WordNetLemmatizer()


class VSM:
    def __init__(self):
        self.index = {}  # declaring inverted index
        self.term_count_doc = {}  # term count in each document
        self.dictionary1 = {}
        self.tfidf = {}        # tfidf for doc
        self.q_tfidf = {}        # tfidf of query
        fptr = open("Stopword-List.txt")  # reading stop word from file
        self.stopwords_list = fptr.readlines()
        self.stopwords_list = [x.rstrip() for x in self.stopwords_list]
        fptr.close()
        self.docs = []
        # This code will open the document file and generate the tokens
        for docId in range(1, 449):
            fptr = open("Abstracts/" + str(docId) + ".txt")
            files = fptr.read().lower()  # lower_casing the words
            files = re.sub('[^A-Za-z]+', ' ', files)
            files = files.replace(".", "").replace("n't", " not").replace("'", "").replace("]", " ").replace("[","").replace(",", " ").replace("?", "").replace("\n", " ").replace("-", " ").replace(":", " ").replace("$"," ")  # word only contain alphabet
            tokens = word_tokenize(files)
            d1 = [lemmatizer.lemmatize(x) for x in tokens]    # apply lemmatization
            rmv_stop = [x for x in d1 if x not in self.stopwords_list]      # remove stopwords
            self.dictionary1[docId] = rmv_stop
            self.docs.append(docId)
            fptr.close()

    def create_index(self):                # creating index
        for i in self.dictionary1.keys():
            tc = 0
            for term in self.dictionary1[i]:
                tc += 1
                if term in self.stopwords_list:
                    continue
                if term not in self.index:
                    self.index[term] = {}
                    self.index[term][i] = []
                    self.index[term][i].append(tc)
                else:
                    if i not in self.index[term]:
                        self.index[term][i] = []
                    self.index[term][i].append(tc)
            self.term_count_doc[i] = tc  # total no of terms in an ith doc for ttf

        # f = open("index.txt",'w')
        # f.write(str(self.index))
        # f.close()

    def calculate_tfidf(self, query):             # calculating tfidf of docs
        self.tfidf = {}
        self.q_tfidf = {}
        #f = open("TF_IDF.txt", "w")
        for term in self.index.keys():
            self.tfidf[term] = {}
            # doc frequency
            df = len(self.index[term])
            idf = math.log(448 / df)  # inverse doc frequency

            self.tfidf[term] = []
            for i in self.docs:
                if i in self.index[term].keys():  # if term exist in doc
                    self.tfidf[term].append(round(((len(self.index[term][i])) * idf),4))  # append tfidf of term in the respective doc
                else:
                    self.tfidf[term].append(0)  # if term doesn't exist in doc
        for term in self.index.keys():
            self.q_tfidf[term] = {}
            df = len(self.index[term])
            idf = math.log(448 / df)  # inverse doc frequency
            self.q_tfidf[term] = []
            if term in query:
                self.q_tfidf[term].append(round(((query.count(term))*idf), 4))  # tfidf of the query
            else:
                self.q_tfidf[term].append(0)
        a_file = open("index.json", "w")
        json.dump(self.index, a_file)
        a_file.close()
        # f.write(str(self.tfidf))
        # f.close()

    def generate_query_vector(self,query):            # generate query vector
        file = pathlib.Path("index.json")
        if file.exists():                        # index is already created
            a_file = open("index.json", "r")
            self.index = a_file.read()
            self.index = json.loads(self.index)
            for term in self.index.keys():
                self.q_tfidf[term] = {}
                df = len(self.index[term])
                idf = math.log(448 / df)  # inverse doc frequency
                self.q_tfidf[term] = []
                if term in query:
                    self.q_tfidf[term].append(round(((query.count(term)) * idf), 4))  # tfidf of the query
                else:
                    self.q_tfidf[term].append(0)

            query_vector = []
            for term in self.q_tfidf.keys():
                query_vector.append(self.q_tfidf[term])
            print("true")
            return query_vector
        else:                               # if index is not created
            query_vector = []
            for term in self.q_tfidf.keys():
                query_vector.append(self.q_tfidf[term])

            return query_vector

    def generate_doc_vector(self, query):
        file = pathlib.Path("doc_vector.json")
        if file.exists():                 # if doc vector is already exist
            a_file = open("doc_vector.json", "r")
            doc_vector = a_file.read()
            doc_vector = json.loads(doc_vector)
            return doc_vector
        else:                      # if doc vector is not exist
            self.create_index()
            self.calculate_tfidf(query)
            doc_vector = {}
            for dno in range(len(self.docs)):
                doc_vector[str(dno)] = []
                for term in self.tfidf.keys():
                    doc_vector[str(dno)].append(self.tfidf[term][dno])
            a_file = open("doc_vector.json", "w")
            json.dump(doc_vector, a_file)
            a_file.close()
            return doc_vector

    def cosine_similarity(self, doc, que):
        cosine = []
        for i in doc.keys():
            out = dot(doc[i], que) / (norm(doc[i]) * norm(que))       # calculate cosine similarity

            if out > 0.05:         # if cosine > alpha
                cosine.append(int(i)+1)
        return cosine

    def processing(self,query):
        query = re.sub('[^A-Za-z]+', ' ', query)  # applying regex to query
        query = query.lower()
        query = word_tokenize(query)
        for t in range(0, len(query)):
            query[t] = lemmatizer.lemmatize((query[t]))
        p = self.generate_doc_vector(query)
        q = self.generate_query_vector(query)
        result = self.cosine_similarity(p, q)
        return result


def Search():
    v = VSM()
    # initialize object of vsm retrieval class
    output.delete("1.0", "end")
    query = str(query_text.get("1.0", END))             # store query from the search box
    result = v.processing(query)

    output.insert(END, 'The Retrieved Documents are ==>  ' + str(result) + ' \n')   # displaying result set



root = tk.Tk()              # onward code is for GUI
root.title('Vector space Model Search Engine')
root.geometry('700x400')
root.configure(bg='grey')
img = ImageTk.PhotoImage(Image.open("engine.jpg"))
panel = Label(root, image=img)   # inserting image
panel.pack()

query_text = Text(root, height=2, width=82)     # Search box
query_text.insert(END, 'Enter your Query.........')
query_text.pack()

search_button = Button(root, height=2, width=15,bg="black", fg="white" , text="Search", command=Search)
search_button.pack()                # button

output = Text(root, height=5,
              width=82,
              bg="light yellow")            # output text
output.pack()

root.mainloop()


