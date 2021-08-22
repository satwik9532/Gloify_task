
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
from PIL import Image
import cv2
from urllib.request import urlretrieve,urlopen,Request
import requests
from bs4 import  BeautifulSoup
from skimage.io import imread
from googlesearch import search

@st.cache()
def load_model():
    model=MobileNetV2()
    return model
# rachit
st.title("welcome to Image Searcher")

st.write("Select Search option from below drop down option")
display=('Search through image','Search through text','normal search')
options = list(range(len(display)))
value = st.selectbox("select search option", options, format_func=lambda x: display[x])


if value==0:
    upload = st.file_uploader(label='Upload the Image')
    if upload is not None:

        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
        img = Image.open(upload)
        st.image(img,caption='Uploaded Image',width=300)
        model = load_model()
        if st.button('Search'):
            st.write("Result:")
            x = cv2.resize(opencv_image,(224,224))
            x = np.expand_dims(x,axis=0)
            x = preprocess_input(x)
            y = model.predict(x)
            label = decode_predictions(y)
            url='https://www.google.com/search?q='+label[0][0][1]+'&rlz=1C1CHBD_enIN914IN914&sxsrf=ALeKk01fX0d1_WW8HEFvKc4HV34HFFbsCQ:1618806932668&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjMz7nhvYnwAhUMbn0KHWSPABsQ_AUoAXoECAEQAw&biw=1455&bih=688'
            r=requests.get(url)
            r=r.content
            soup=BeautifulSoup(r,'html.parser')
            link=soup.find_all('img')
            link.pop(0)
            for i in link:
                img=imread(i['src'])
                st.image(img,width=300)
            # print the classification
            
            out = label[0][0]
            st.title(out[1])   
   
          
elif value==1:
    i=st.text_input('enter the word to searching image')
    if st.button('Search'):
        url='https://www.google.com/search?q='+i+'&rlz=1C1CHBD_enIN914IN914&sxsrf=ALeKk01fX0d1_WW8HEFvKc4HV34HFFbsCQ:1618806932668&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjMz7nhvYnwAhUMbn0KHWSPABsQ_AUoAXoECAEQAw&biw=1455&bih=688'
        r=requests.get(url)
        r=r.content
        soup=BeautifulSoup(r,'html.parser')
        link=soup.find_all('img')
        link.pop(0)
        for i in link:
            img=imread(i['src'])
            st.image(img)
else:
    i=st.text_input('Enter to word to search releted query')
    for j in search(i,  num=10, stop=10, pause=1):
        st.write(j)    

      
           


