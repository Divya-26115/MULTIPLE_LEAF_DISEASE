import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
import telepot
bot=telepot.Bot("8114463941:AAE7XGZ68QqGT4MEHJOOrCpSfTA4OO7NXY4")
ch_id="8231601134"
discls = ['Tomato_yellow_curl_leaf','Tomato_spectoria','Tomato_leafmold','Tomato_bacterial_spot',
                                'Banana_sigatoka','Banana_pestalotiopsis','Banana_cordana',
                                'Paddy_bacterial','Paddy_brownspot','Paddy_Leafsmut',
                                'fussarium_wilt','Target_spot','Powdery_Mildew','Bacterial_Blight','Army_worm_cotton_leaf','Aphids_cotton_leaf']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog.html')
def demo():
    return render_template('userlog.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
            # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'leafdisease-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
              verifying_data = []
              for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append((img, img_num))  # use tuple instead of list
                return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 19, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        rem=" "
        rem1=" "
        str_label=" "
        accuracy=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict(np.array([data]))[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Aphids_cotton_leaf'
                print("The predicted image of the Aphids_cotton_leaf is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the Aphids_cotton_leaf is with a accuracy of {}%".format(model_out[0]*100)
                rem = "The remedies for Bacterial Spot are:\n\n "
                rem1 = [" Methyl demeton 25 EC 500ml",
                "Dimethoate 30 EC 500ml",
                "Acetamiprid 20% SP 50 g",
                "Azadirachtin 0.03% EC 500 ml",
                "Buprofezin 25% SC1000 ml",
                "Carbosulfan 25%DS 60g/kg of seed"]
            
            elif np.argmax(model_out) == 1:
                str_label = 'Army_worm_cotton_leaf'
                print("The predicted image of the Army_worm_cotton_leaf is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the Army_worm_cotton_leaf is with a accuracy of {}%".format(model_out[1]*100)
                rem = "The remedies for Yellow leaf curl virus are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates", 
                "carbametes during the seedliing stage.", "Use copper fungicites"]
                
                
            elif np.argmax(model_out) == 2:
                str_label = 'Bacterial_Blight'
                print("The predicted image of the Bacterial_Blight is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the Bacterial_Blight is with a accuracy of {}%".format(model_out[2]*100)
                rem = "The remedies for spectoria are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]
                
                
            elif np.argmax(model_out) == 3:
                str_label = 'Healthy_cotton'
                print("The predicted image of the Healthy_leaf is with a accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the Healthy_leaf is with a accuracy of {}%".format(model_out[3]*100)
                

                
            elif np.argmax(model_out) == 4:
                str_label = 'Powdery_Mildew'
                print("The predicted image of the Powdery_Mildew is with a accuracy of {} %".format(model_out[4]*100))
                accuracy="The predicted image of the Powdery_Mildew is with a accuracy of {}%".format(model_out[4]*100)
                rem = "The remedies for Leafmold are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
                


            elif np.argmax(model_out) == 5:
                str_label = 'Target_spot'
                print("The predicted image of the Target_spot is with a accuracy of {} %".format(model_out[5]*100))
                accuracy="The predicted image of the Target_spot is with a accuracy of {}%".format(model_out[5]*100)
                rem = "The remedies for Leafmold are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]

            elif np.argmax(model_out) == 6:
                str_label = 'fussarium_wilt'
                print("The predicted image of the fussarium_wilt is with a accuracy of {} %".format(model_out[6]*100))
                accuracy="The predicted image of the fussarium_wilt is with a accuracy of {}%".format(model_out[6]*100)
                rem = "The remedies for fussarium_wilt are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]

            elif np.argmax(model_out) == 7:
                str_label = 'Paddy_bacterial'
                print("The predicted image of the Paddy_bacterial is with a accuracy of {} %".format(model_out[7]*100))
                accuracy="The predicted image of the Paddy_bacterial is with a accuracy of {}%".format(model_out[7]*100)
                rem = "The remedies for Paddy_bacterial are: "
                rem1 = [" Use disease-resistant varieties.",
                "Crop rotation.",
                "Cultural practices.",
                "Chemical control"]
                
            elif np.argmax(model_out) == 8:
                str_label = 'Paddy_brownspot'
                print("The predicted image of the Paddy_brownspot is with a accuracy of {} %".format(model_out[8]*100))
                accuracy="The predicted image of the Paddy_brownspot is with a accuracy of {}%".format(model_out[8]*100)
                rem = "The remedies for Paddy_brownspot are: "
                rem1 = [" Crop rotation.",
                "Cultural practices",
                "Fungicide application.",
                    "Seed treatment."]
                
            elif np.argmax(model_out) == 9:
                str_label = 'Paddy_Leafsmut'
                print("The predicted image of the Paddy_Leafsmut is with a accuracy of {} %".format(model_out[9]*100))
                accuracy="The predicted image of the Paddy_Leafsmut is with a accuracy of {}%".format(model_out[9]*100)
                rem = "The remedies for Paddy_Leafsmutare: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
            elif np.argmax(model_out) == 10:
                str_label = 'Banana_cordana'
                print("The predicted image of the Banana_cordana is with a accuracy of {} %".format(model_out[10]*100))
                accuracy="The predicted image of the Banana_cordana is with a accuracy of {}%".format(model_out[10]*100)
                rem = "The remedies for Banana_cordana are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
            elif np.argmax(model_out) == 11:
                str_label = 'Banana_healthy'
                print("The predicted image of the Banana_healthy is with a accuracy of {} %".format(model_out[11]*100))
                accuracy="The predicted image of the Banana_healthy is with a accuracy of {}%".format(model_out[11]*100)

            elif np.argmax(model_out) == 12:
                str_label = 'Banana_pestalotiopsis'
                print("The predicted image of the Banana_pestalotiopsis is with a accuracy of {} %".format(model_out[12]*100))
                accuracy="The predicted image of the Banana_pestalotiopsis is with a accuracy of {}%".format(model_out[12]*100)
                rem = "The remedies for Banana_pestalotiopsis are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]

            elif np.argmax(model_out) == 13:
                str_label = 'Banana_sigatoka'
                print("The predicted image of the Banana_sigatoka is with a accuracy of {} %".format(model_out[13]*100))
                accuracy="The predicted image of the Banana_sigatoka is with a accuracy of {}%".format(model_out[13]*100)
                rem = "The remedies for Banana_sigatoka are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]

            elif np.argmax(model_out) == 14:
                str_label = 'Tomato_bacterial_spot'
                print("The predicted image of the Tomato_bacterial_spot is with a accuracy of {} %".format(model_out[14]*100))
                accuracy="The predicted image of the Tomato_bacterial_spot is with a accuracy of {}%".format(model_out[14]*100)
                rem = "The remedies for Tomato_bacterial_spot are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
            elif np.argmax(model_out) == 15:
                str_label = 'Tomato_healty'
                print("The predicted image of the Tomato_healty is with a accuracy of {} %".format(model_out[15]*100))
                accuracy="The predicted image of the Tomato_healty is with a accuracy of {}%".format(model_out[15]*100)
                
            elif np.argmax(model_out) == 16:
                str_label = 'Tomato_leafmold'
                print("The predicted image of the Tomato_leafmold is with a accuracy of {} %".format(model_out[16]*100))
                accuracy="The predicted image of the Tomato_leafmold is with a accuracy of {}%".format(model_out[16]*100)
                rem = "The remedies for Leafmold are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
            elif np.argmax(model_out) == 17:
                str_label = 'Tomato_spectoria'
                print("The predicted image of the Tomato_spectoria is with a accuracy of {} %".format(model_out[17]*100))
                accuracy="The predicted image of the Tomato_spectoria is with a accuracy of {}%".format(model_out[17]*100)
                rem = "The remedies for spectoria are: "
                rem1 = [" Monitor the field, handpick diseased plants and bury them.",
                "Use sticky yellow plastic traps.", 
                "Spray insecticides such as organophosphates",
                "carbametes during the seedliing stage.",
                "Use copper fungicites"]
                
            elif np.argmax(model_out) == 18:
                str_label = 'Tomato_yellow_curl_leaf'
                print("The predicted image of the Tomato_yellow_curl_leaf is with a accuracy of {} %".format(model_out[18]*100))
                accuracy="The predicted image of the Tomato_yellow_curl_leaf is with a accuracy of {}%".format(model_out[18]*100)
                rem = "The remedies for fussarium_wilt are: "
                rem1 = [" Monitor the field, remove and destroy infected leaves.",
                "Treat organically with copper spray.",
                "Use chemical fungicides,the best of which for tomatoes is chlorothalonil."]
                
            try:
                if str_label in discls:
                    
                    bot.sendMessage(ch_id,str_label)
                    bot.sendMessage(ch_id,rem1)
                    bot.sendPhoto(ch_id,photo=open("test/"+fileName,'rb'))
                    
                    
                elif str_label in ['Tomato_healty','Banana_healthy','Healthy_cotton']:
                    
                    bot.sendMessage(ch_id,str_label)
                
                    bot.sendPhoto(ch_id,photo=open("test/"+fileName,'rb'))
            except Exception as e:
                print(f"Telegram message not Send {e}")
            
                
            try:
                if str_label in discls:
                    from serial_test import Send
                    Send("A")
                elif str_label in ['Tomato_healty','Banana_healthy','Healthy_cotton']:
                    from serial_test import Send
                    Send("B")
     
            except Exception as e:
                print(f"hardware not connected{e}")
    
    

            return render_template('results.html', status=str_label,accuracy=accuracy,remedie=rem, remedie1=rem1, 
                                   ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,
                                   ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
                                   ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
                                   ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",
                                   ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
    
        return render_template('index.html')
@app.route('/logout')
def logout():
   return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
