import os
from flask import Flask, render_template, request, redirect, url_for, abort,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import spacy
from collections import defaultdict
import re
import pickle
import numpy as np
from spacy import load
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv']

app.config['PATH_NAME'] = os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def index():
    return "<html><head><title>File Upload</title></head><body><h1>File Upload</h1><form method=\"POST\" action=\"\" enctype=\"multipart/form-data\"> <p><input type=\"file\" name=\"csv_file\" accept=\".csv\"></p><p><input type=\"submit\" value=\"Submit\"></p></form></body></html>"

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['csv_file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return "<html><head><title>L\'extension n\'est pas bonne File Upload</title></head><body><h1>L\'extension n\'est pas bonne</h1><h1>File Upload</h1><form method=\"POST\" action=\"\" enctype=\"multipart/form-data\"> <p><input type=\"file\" name=\"csv_file\" accept=\".csv\"></p><p><input type=\"submit\" value=\"Submit\"></p></form></body></html>"            
        data = pd.read_csv(uploaded_file, sep=",")
        if 'Title' not in data.columns:
            print('le csv ne contient pas de colonne Title')
            return "<html><head><title>le csv ne contient pas de colonne Title File Upload</title></head><body><h1>le csv ne contient pas de colonne Title</h1><h1>File Upload</h1><form method=\"POST\" action=\"\" enctype=\"multipart/form-data\"> <p><input type=\"file\" name=\"csv_file\" accept=\".csv\"></p><p><input type=\"submit\" value=\"Submit\"></p></form></body></html>"
        if 'Body' not in data.columns:
            print('le csv ne contient pas de colonne Body')
            return "<html><head><title>le csv ne contient pas de colonne Body File Upload</title></head><body><h1>le csv ne contient pas de colonne Body</h1><h1>File Upload</h1><form method=\"POST\" action=\"\" enctype=\"multipart/form-data\"> <p><input type=\"file\" name=\"csv_file\" accept=\".csv\"></p><p><input type=\"submit\" value=\"Submit\"></p></form></body></html>"
            ####       
        data=data[['Title','Body']]
        data_dict2=data.set_index('Title').to_dict('index')    
        for k in data_dict2:
            #data_dict2[k]['Body2']=re.sub('\<\/a','</p',data_dict2[k]['Body'])

            #on passe le corps du texte en minuscule et on le copie dans Body2
            data_dict2[k]['Body2']=data_dict2[k]['Body'].lower()
            #on remplace les balises <a> en balises <p>
            #value=re.sub('\<a','<p',data_dict2[k]['Body2'])
            #data_dict2[k]['Body2']=value
            #data_dict2[k]['Body2']=re.sub('\<\/a','</p',data_dict2[k]['Body2'])
            soup = BeautifulSoup(data_dict2[k]['Body2'])
            data_dict2[k]['Body2']=[]
    
            #on remplace les balises pre en p tout en gardant la balise <code>
            for p in soup.find_all('pre'):
                n = BeautifulSoup('<p><code>%s</code></p>' % p.string)
                p.replace_with(n.body.contents[0])    
            
            #idem pour la balise h1
            for p in soup.find_all('h1'):
                n = BeautifulSoup('<p>%s</p>' % p.string)
                p.replace_with(n.body.contents[0]) 
            for p in soup.find_all('h2'):
                n = BeautifulSoup('<p>%s</p>' % p.string)
                p.replace_with(n.body.contents[0])     #on recupere la liste comprises entre des balises p dans Body2
            for p in soup.find_all('p') :
                data_dict2[k]['Body2'].append(str(p))

        #on copie body2 dans body3    
        for k in data_dict2:
            data_dict2[k]['Body3']=data_dict2[k]['Body2'].copy()

        #on boucle sur les questions
        for k in data_dict2:    
            #on boucle sur la liste de p extraite
            for index, value in enumerate(data_dict2[k]['Body3']):        
                #on regarde si on trouve la balise code à l'interieur d'un <p>
                #on la modifie de maniere a fermer le p et a en ouvrir un nouveau
                value = value.replace('<code>', '</p><p><code>') 
                value = value.replace('</code>', '</code></p><p>')
                value = value.replace('<h2>', '</p><p><h2>') 
                value = value.replace('</h2>', '</h2></p><p>')
                try:
                    indice_a=value.index("<a")
                except ValueError:
                    indice_a=-1       
                if indice_a!=3 and indice_a!=-1:
                    value = value.replace('<a', '</p><p><a') 
                    value = value.replace('</a>', '</a></p><p>')
                #on soup
                soup2 = BeautifulSoup(value)
                list_temp=[]    
                #on recupere la liste comprises dans les balises p et on l'insere dans la liste précédente
                for q in soup2.find_all('p') :
                    list_temp.append(str(q))
                #on supprime les <p> vide 
                list_temp=list(filter(("<p></p>").__ne__,list_temp))
                data_dict2[k]['Body3'][index]=list_temp

        #on supprime les elements vide de la liste
        for k in data_dict2:    
            for index, value in enumerate(data_dict2[k]['Body3']):        
                if len(value)==0:
                    del data_dict2[k]['Body3'][index]
            for index, value in enumerate(data_dict2[k]['Body3']):        
                if len(value)==0:
                    del data_dict2[k]['Body3'][index]


        #on boucle sur les questions
        for k in data_dict2:
            
            data_dict2[k]['Body_texte']=""
            data_dict2[k]['Body_texte_code']=""
            for index, value in enumerate(data_dict2[k]['Body3']):
                for index2, value2 in enumerate(data_dict2[k]['Body3'][index]):
                    texte_temp=data_dict2[k]['Body3'][index][index2]
                    #on teste si on est sur un element de code ou de lien
                    try:
                        indice_code=texte_temp.index("<code>")
                    except ValueError:
                        indice_code=-1
                    try:
                        indice_href=texte_temp.index("<a ")
                    except ValueError:
                        indice_href=-1
                        
                    #si on est un element du corps, on sauvegarde dans body_texte et body_texte_code
                    if indice_code==-1 and indice_href==-1:
                        # on gère deux exceptions du texte avant la tokenization
                        #"\'" par "'"
                        texte_temp = texte_temp.replace('()', '() ')
                        texte_temp = texte_temp.replace('''\\\'''','''\'''')
                        #on enleve les balises et les sauts de lignes
                        texte_temp = texte_temp.replace('\n', ' ')
                        texte_temp = texte_temp.replace('<em>', '')
                        texte_temp = texte_temp.replace('</em>', '')
                        texte_temp = texte_temp.replace('<strong>', '')
                        texte_temp = texte_temp.replace('</strong>', '')
                        texte_temp = texte_temp.replace('<br>', '')
                        texte_temp = texte_temp.replace('</br>', '')
                        texte_temp = texte_temp.replace('<br/>', '')
                        texte_temp = texte_temp.replace('</p>', '')
                        texte_temp = texte_temp.replace('<p>', '')
                        texte_temp = texte_temp.replace('<h1>', '')
                        texte_temp = texte_temp.replace('</h1>', '')
                        texte_temp = texte_temp.replace('<h2>', '')
                        texte_temp = texte_temp.replace('</h2>', '')
                        texte_temp = texte_temp.replace('<blockquote>', '')
                        texte_temp = texte_temp.replace('</blockquote>', '')
                        data_dict2[k]['Body_texte']=data_dict2[k]['Body_texte']+" "+texte_temp
                        data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code']+" "+texte_temp
                    #si on est un element du code, on sauvegarde dans body_texte_code            
                    if indice_code!=-1 and indice_href==-1:
                                     
                        #on enleve les balises et les sauts de lignes
                        texte_temp = texte_temp.replace('\n', ' ')
                        texte_temp = texte_temp.replace('<em>', '')
                        texte_temp = texte_temp.replace('</em>', '')
                        texte_temp = texte_temp.replace('<strong>', '')
                        texte_temp = texte_temp.replace('</strong>', '')
                        texte_temp = texte_temp.replace('<br>', '')
                        texte_temp = texte_temp.replace('</br>', '')
                        texte_temp = texte_temp.replace('<br/>', '')
                        texte_temp = texte_temp.replace('</p>', '')
                        texte_temp = texte_temp.replace('<p>', '')
                        texte_temp = texte_temp.replace('<h1>', '')
                        texte_temp = texte_temp.replace('</h1>', '')
                        texte_temp = texte_temp.replace('<h2>', '')
                        texte_temp = texte_temp.replace('</h2>', '')
                        texte_temp = texte_temp.replace('<blockquote>', '')
                        texte_temp = texte_temp.replace('</blockquote>', '')
                        texte_temp = texte_temp.replace('</code>', '')
                        texte_temp = texte_temp.replace('<code>', '')            
                        data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code']+" "+texte_temp    
            data_dict2[k]['Body_texte_code']=data_dict2[k]['Body_texte_code'][1:]
            data_dict2[k]['Body_texte']=data_dict2[k]['Body_texte'][1:]
            
        most_freq=('i', 'the', 'to', 'a', 'is', 'and', 'in', 'this', 'it', 'my', 'of', 'but', 'that', 'have', 'with', 'for', 'error', 'on', 'not', 'am', 'can', 'using', 'when', 'how', 't', 'as', 'm', 'from', 'an', 'be', 'get', 'like', 'do', 'so', 'if', 'what', 'at', 'any', 'use', 'want', 'or','s', 'tried', 'there', 'trying', 'here','following', 'which', 'are', 'me', 'all', 'run', 'you', 'some', 'would', 'way', 'by', 'new', 'problem', 'one', 'work', 'no', 've', 'need', 'was', 'help', 'then', 'also', 'below',  'getting', 'know', 'has', 'same', 'only', 'just', 'does', 'working', 'now','will', 'should', 'build', 'after')
        nltk.download('stopwords')
        # On créé notre set de stopwords final qui cumule ainsi les 100 mots les plus fréquents du corpus ainsi que l'ensemble de stopwords par défaut présent dans la librairie NLTK
        sw = set()
        sw.update(most_freq)
        sw.update(tuple(nltk.corpus.stopwords.words('english')))
        tuple_sw_ajout=('I','ca','\'s','could', 'know','run','want','use','try','err','error','errror','like','question','issue','example','solution','finally','follow','look','think','thank','make','code','answer','understand','thing','happen','say','sure','really','good','hello','tell','little','warning','late','fix','change')
        sw.update(tuple_sw_ajout)
        #ajout du lemmatizer
        nlp= en_core_web_sm.load()
        lemmatizer = nlp.get_pipe("lemmatizer")

        #ajout de la racinisation nlp 

        suffixes = list(nlp.Defaults.suffixes)
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        nlp.tokenizer.suffix_search = suffix_regex.search

        prefixes = list(nlp.Defaults.prefixes)
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search
         

        #on tokenize
        import unicodedata
     
        body_sans_stopwords_lemme_stem_sans_nltk_speller = defaultdict(list)
        for question in data_dict2:
            #on enleve les accents
            data_dict2[question]['Body_texte']=str(unicodedata.normalize('NFD', data_dict2[question]['Body_texte']).encode('ascii', 'ignore'))[2:-1] 
            data_dict2[question]['Body_texte']=data_dict2[question]['Body_texte'].lower()
            doc=nlp(data_dict2[question]['Body_texte'])
            tokens=[token.lemma_ for token in doc]
            #on enleve les mots compris dans sw
            body_sans_stopwords_lemme_stem_sans_nltk_speller[question] += [w for w in tokens if (not w in list(sw)) ]

        for question in body_sans_stopwords_lemme_stem_sans_nltk_speller:
            index=0
            while index<len(body_sans_stopwords_lemme_stem_sans_nltk_speller[question]):
                #drapeau pour dire qu'on a effectué une opération et donc que l'on incrémente pas l'index pour retester
                drap=0
                value=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                #on supprime si la valeur est vide
                if value=="":
                    del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                    drap=1
                
                #on regarde si le mot suivant n'est pas "(" et celui d'après ")"
                #si c'est le cas, on agrege "()" au premier mot et on supprime les deux suivants
                if (index+1)<=(len(body_sans_stopwords_lemme_stem_sans_nltk_speller[question])-1) and drap==0:
                    if body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index+1]==")" and value=="(":
                        body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index-1]=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index-1]+"()"
                        del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index+1]
                        del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                        drap=1
                #on supprime "n't" ou les verbes        
                if ("n\'t" in value or "\'m" in value or "\'ve" in value or '\'re' in value or '\'d' in value or value=="it\\" or value=="run" ) and drap==0:
                    del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                    drap=1      
                #on remplace les \' par '
                try:
                    indice=value.index('''\\\'''')
                except ValueError:
                    indice=-1
                if indice!=-1 and drap==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].replace('''\\\'''','''\'''')
                    drap=1
                #on supprime les années
                if (re.search("20[0-9]{2}",value) is not None) and len(value)==4 and drap==0:
                    del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                    drap=1
                #on remplace pe par ping
                if value=="pe" and drap==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]="ping"
                    drap=1
                #on corrige instal
                if value=="instal" and drap==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]="install"
                    drap=1
                #on corrige jupiter
                if value=="jupiter" and drap==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]="jupyter"
                    drap=1
                #on supprime le . s'il est en début ou en fin de mot
                try:
                    indice=value.index(".")
                except ValueError:
                    indice=-1
                if (indice==0 or indice==len(value)-1) and value!=".com" and value!=".org" and value!=".fr" and value!=".net" and value!=".exe" and value!=".ini" and value!=".bat" and value!=".py" and value!=".js" and value!=".net-core" and drap==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index] = value.replace('.', '')
                    drap=1
                
                #on supprime le ? s'il est en début ou en fin de mot
                try:
                    indice=value.index("?")
                except ValueError:
                    indice=-1
                if (indice==0 or indice==len(value)-1) and drap==0:
                    body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index] = value.replace('?', '')
                    drap=1
                
                #on teste si l'expression ne contient pas les mots suivants mais contient un . et pas de ()
                #on teste aussi que cela soit pas une adresse ip ou un numero de version
                try:
                    indice=value.index(".com")
                except ValueError:
                    indice=-1
                try:
                    indice1=value.index(".org")
                except ValueError:
                    indice1=-1
                try:
                    indice2=value.index(".fr")
                except ValueError:
                    indice2=-1        
                try:
                    indice2=value.index(".net")
                except ValueError:
                    indice2=-1        
                try:
                    indice3=value.index(".exe")
                except ValueError:
                    indice3=-1        
                try:
                    indice4=value.index(".ini")
                except ValueError:
                    indice4=-1        
                try:
                    indice5=value.index(".bat")
                except ValueError:
                    indice5=-1        
                try:
                    indice6=value.index(".py")
                except ValueError:
                    indice6=-1        
                try:
                    indice7=value.index(".js")
                except ValueError:
                    indice7=-1        
                try:
                    indice8=value.index(".net-core")
                except ValueError:
                    indice8=-1        
                try:
                    indice9=value.index("()")
                except ValueError:
                    indice9=-1         
                
                try:
                    indice10=value.index(".")
                except ValueError:
                    indice10=-1        
                if indice==-1 and indice1==-1 and indice2==-1 and indice3==-1 and indice4==-1 and indice5==-1 and indice6==-1 and indice7==-1 and indice8==-1 and indice9==-1 and indice10!=-1 and re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",value) is None and re.match(r"\d{1,3}\.\d{1,3}",value) is None and re.match(r"\d{1,3}\.x",value) is None and re.match(r"\d{1,3}\.X",value) is None and re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}",value) is None and drap==0:    
                    #il faut donc séparer les mots par le .
                    list_temp=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].split(".")
                    #on crée une nouvelle liste que l'on insere dans la liste mere
                    for index2, value2 in enumerate(list_temp):
                        if index2==0:
                            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=list_temp[index2]
                        else:
                            body_sans_stopwords_lemme_stem_sans_nltk_speller[question].insert(index+index2, list_temp[index2])
                    drap=1
                
                #on teste si l'expression contient un ?
                try:
                    indice=value.index("?")
                except ValueError:
                    indice=-1        
                if  indice!=-1 and drap==0:    
                    #il faut donc séparer les mots par le ?
                    list_temp=body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index].split("?")
                    #on crée une nouvelle liste que l'on insere dans la liste mere
                    for index2, value2 in enumerate(list_temp):
                        if index2==0:
                            body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]=list_temp[index2]
                        else:
                            body_sans_stopwords_lemme_stem_sans_nltk_speller[question].insert(index+index2, list_temp[index2])
                    drap=1
           
                #on teste si le mot est composé d'espace
                y=""
                x=0
                while x<20:
                    if value==y and drap==0:
                        del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                        drap=1
                    y=y+" "
                    x=x+1
                #on teste si le mot est de longueur 1 et est non alphanumerique
                if (re.search("[a-zA-Z0-9]",value) is None or value=="," or value=="i") and len(value)==1 and drap==0:
                    del body_sans_stopwords_lemme_stem_sans_nltk_speller[question][index]
                    drap=1
                
                #si on a effectué aucune modification, on peut augmenter l'index
                if drap==0:
                    index=index+1           
         
        df_tags_matrix= pd.DataFrame.from_dict(data_dict2, orient='index')
        df_tags_matrix['question']=df_tags_matrix.index
        list_body_model=[]
        list_question_model=[]
        for question, value in body_sans_stopwords_lemme_stem_sans_nltk_speller.items():
            body=" ".join(value)
            data_dict2[question]['Body_token']=body
            list_body_model.append(body)
            list_question_model.append(question)
            
        
        #max_features=nombre de mots que l'on va garder dans le vocabulaire
        
        data_fname = 'vector_body.pkl'
        with open(os.path.join(app.config['PATH_NAME'], data_fname), "rb") as tf:
            tf_vectorizer_body = pickle.load(tf)
        tf_body=tf_vectorizer_body.transform(list_body_model)
        
        
        df_tags_matrix_multiclass_mlp_predict=df_tags_matrix['question']
        df_tags_matrix_multiclass_mlp_predict=df_tags_matrix_multiclass_mlp_predict.reset_index(drop=True)
        print("a")
        list_tag_brute=['python', 'tensorflow', 'flutter', 'flutter-layout', 'javascript', 'node.js', 'google', 'c#', 'asp.net-core', 'angular', 'angular6', 'jquery', 'reactjs', 'typescript', 'git', 'java', 'android', 'android-studio', 'gradle', 'kotlin', 'pandas', 'docker', 'apache', 'php', 'mysql', 'ubuntu', 'spring', 'python-3.x', 'spring-boot', 'dart', 'html', 'sql', 'react-native', 'css', 'asp.net', 'numpy', 'material-ui', 'vue.js', 'laravel', 'amazon-web-services', 'amazon', 'kubernetes', 'dataframe', 'c++', 'webpack', 'visual-studio', 'keras', 'jestjs', '.net-core', 'swift', 'arrays', 'xcode', 'angular-material', 'ios', 'firebase', 'json', 'vuejs2', 'docker-compose', 'django', 'react-hooks', 'visual-studio-code', 'macos', 'npm', 'bootstrap-4', 'windows', 'jupyter', 'linux', 'selenium']
        data_fname = 'mlp_tf_multiclass.pkl'
        with open(os.path.join(app.config['PATH_NAME'], data_fname), "rb") as tf:
            model = pickle.load(tf)
        print("b")
        y_predict=model.predict_proba(tf_body)

        temp=pd.DataFrame(y_predict,  columns=["prob_prediction_1_mlp_multiclass_tf_tag_"+i for i in list_tag_brute] )
        df_tags_matrix_multiclass_mlp_predict=pd.concat([df_tags_matrix_multiclass_mlp_predict,temp],axis=1)
        x=0
        while x<len(list_tag_brute):       
            df_tags_matrix_multiclass_mlp_predict[list_tag_brute[x]]=np.where(df_tags_matrix_multiclass_mlp_predict['prob_prediction_1_mlp_multiclass_tf_tag_'+list_tag_brute[x]]>=0.8,1,0)
            del df_tags_matrix_multiclass_mlp_predict['prob_prediction_1_mlp_multiclass_tf_tag_'+list_tag_brute[x]]
            x=x+1
        df_tags_matrix_multiclass_mlp_predict=df_tags_matrix_multiclass_mlp_predict.set_index('question')
        df_dict = dict(
            list(
                df_tags_matrix_multiclass_mlp_predict.groupby(df_tags_matrix_multiclass_mlp_predict.index)
            )
        )
        df_dict2=dict()
        for k, v in df_dict.items():               # k: name of index, v: is a df
            check = v.columns[(v == 1).any()]
            df_dict2[k]=list()
            if len(check) > 0:
               df_dict2[k]=check.to_list()
        
    #        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return jsonify(df_dict2)
    