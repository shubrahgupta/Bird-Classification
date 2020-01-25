import os
import pandas as pd
from glob import glob
import numpy as np

from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure


def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = '/home/shubrah/ml/pad/train/' + name

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = '/home/shubrah/ml/pad/test/' + name
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
def create_df(folder,some):
    folder1=[]
    for i in range(len(folder)):
        folder1.append([folder[i],folder[i].split('/')[4],'/home/shubrah/ml/pad/'+some+'/'+folder[i].split('/')[5].split('.')[0]+'.png'])
    a=pd.DataFrame(folder1,columns=['location','class','filename'])
    return a
    
    
    
extract=['/home/shubrah/bird/INSC_1/7b050ddd-94fa-4e7e-8b4b-14651e014520.wav',
 '/home/shubrah/bird/MAWT_1/e6c111e9-8406-4b6c-ba7d-c506ba25c176.wav',
 '/home/shubrah/bird/MAWT_1/293c5041-6039-46c0-8a4c-0d0ac4cac275.wav',
 '/home/shubrah/bird/UNID/c2ac58a2-1b74-47fe-96f4-97299b7dad5f.wav',
 '/home/shubrah/bird/AFBB_1/a19a84a1-8889-4a36-807f-6b4786dd70ea.wav',
 '/home/shubrah/bird/MAWT_s/29f35de3-70ef-41fd-aaf4-36d1a09da895.wav',
 '/home/shubrah/bird/MAWT_s/8f90959b-b0f5-431f-911a-eb2e8d248491.wav',
 '/home/shubrah/bird/MAWT_s/17eabff4-6004-41aa-b679-80ff7103a35b.wav',
 '/home/shubrah/bird/MAWT_s/775204cf-4f14-4e70-b402-163fef80c0d4.wav',
 '/home/shubrah/bird/WHCB_s/271eb2a9-0b30-46d7-a316-287740447b5b.wav',
 '/home/shubrah/bird/INSECT_1/7c25b229-9fca-440f-a694-5a2cabfd82ad.wav',
 '/home/shubrah/bird/INSECT_1/a1e0b34c-8188-4fd2-852f-9d4513e8cbe9.wav',
 '/home/shubrah/bird/INSECT_1/4f95ccc0-e48d-45d1-a13f-4fb81428f364.wav',
 '/home/shubrah/bird/REWB_1/a027b0ed-ff44-408e-88d2-8520b886754b.wav',
 '/home/shubrah/bird/REWB_1/dcf9c3e0-6070-4828-87dd-a063166ce612.wav',
 '/home/shubrah/bird/REWB_1/66354a91-c22d-4886-b65b-af0f1b92b9bd.wav',
 '/home/shubrah/bird/REWB_1/aa8406b2-732b-46c5-8c57-e42b519f131c.wav',
 '/home/shubrah/bird/REWB_1/30327c1e-54e5-4ec9-9195-6c2a057fb012.wav',
 '/home/shubrah/bird/REWB_1/338f25d0-afb1-42f9-9d66-98aa54af9348.wav',
 '/home/shubrah/bird/YEBB_1/23cc7059-ad02-4b7d-905a-edb8503d4f69.wav',
 '/home/shubrah/bird/INSB_1/1e526283-de1b-47f4-a6c8-1fad0fa5d714.wav',
 '/home/shubrah/bird/INSB_1/c9a731b0-699b-49c8-8b87-d48ad45d8288.wav',
 '/home/shubrah/bird/LABC_1/80404893-0cbb-42b3-b9c1-f95e069fa90a.wav',
 '/home/shubrah/bird/LABC_2/d244afaf-7fa4-4c34-8ee7-d69682d9c2f8.wav',
 '/home/shubrah/bird/GRTD_s/22ba0fdb-42d6-471f-ad57-603bfa139b1e.wav']

for i in range(len(extract)):
    create_spectrogram(extract[i],extract[i].split('/')[5].split('.')[0])

traindf=create_df(extract,'train')
    
from keras_preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/home/shubrah/ml/pad/train/",
    x_col="filename",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="/home/shubrah/ml/pad/train/",
    x_col="filename",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(traindf['class'])), activation='softmax'))
model.name = '1st model'
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    epochs=100
)

#Evaluate GEnerator
model.evaluate_generator(generator=valid_generator)

extract1=['/home/shubrah/bird2/INSECT_6/dcb9f121-78c5-4ac6-b7dc-3baf311434b6.wav',
 '/home/shubrah/bird2/FTBU_1/22437d9c-4ae6-4f0d-a680-22d89b084ada.wav',
 '/home/shubrah/bird2/PUTB_s/ea560087-da67-44cd-8662-15970b680de2.wav',
 '/home/shubrah/bird2/PUTB_s/82fe26fa-bd03-407c-92bd-b6ae3c5fdb06.wav',
 '/home/shubrah/bird2/PUTB_s/bcf5e55c-7609-47aa-9403-b69183408f8c.wav',
 '/home/shubrah/bird2/PUTB_s/e5bf2df0-1b2f-48e1-9a95-2c3637e26091.wav',
 '/home/shubrah/bird2/PUTB_s/b5c3261f-10a9-4e0a-b696-0bc7b6a61ed8.wav',
 '/home/shubrah/bird2/PUTB_s/6c5e5031-956d-45fd-b72e-46281de98b01.wav',
 '/home/shubrah/bird2/PUTB_s/e2580614-071f-4d63-8189-adafe5555dec.wav',
 '/home/shubrah/bird2/PUTB_s/d3008d7b-9352-4579-afca-6a78dc001efc.wav',
 '/home/shubrah/bird2/PUTB_s/2f82181f-1e22-4bd2-b566-5a254dd3da66.wav',
 '/home/shubrah/bird2/LABC_s/b6ae5123-8c01-4a5a-845a-4d3d54c824b5.wav',
 '/home/shubrah/bird2/INSCB_s/4bb8576a-e5cd-432e-9135-e9665a284226.wav',
 '/home/shubrah/bird2/SQUIRREL_1/a1a38c9b-3e0a-4930-8f22-7497698fec28.wav',
 '/home/shubrah/bird2/FLTB_1/b6a8ab4d-8488-472d-99fc-08efee5ff032.wav',
 '/home/shubrah/bird2/FLTB_1/8b7d996b-894f-433f-ae2b-a5c1a9b277e7.wav',
 '/home/shubrah/bird2/FLTB_1/8e03024b-cf0e-486d-8e1a-90e2d8397eb6.wav',
 '/home/shubrah/bird2/FLTB_1/86080546-012f-460c-824d-273abfe1f394.wav',
 '/home/shubrah/bird2/FLTB_1/b2fc96d4-8d7c-482e-8087-57b228b218bf.wav',
 '/home/shubrah/bird2/FLTB_1/c4ba5fcc-2bc9-46b6-9808-180abb49cd91.wav',
 '/home/shubrah/bird2/FLTB_1/494d1cc4-1af8-4fc2-8ab2-42586b9ba1fe.wav',
 '/home/shubrah/bird2/FLTB_1/dc97b12e-1d0b-455e-b24d-05b1154d29cb.wav',
 '/home/shubrah/bird2/FLTB_1/b0e130a4-980b-4741-b0dd-2f9c649c1be6.wav',
 '/home/shubrah/bird2/FLTB_1/1b3a3c6b-605f-465e-a59b-67ab394fc790.wav',
 '/home/shubrah/bird2/FLTB_1/d177fd18-068e-4bf4-bbee-8328a2a19419.wav',
 '/home/shubrah/bird2/INSECT_2/a5e0d187-8d49-4b9c-9777-e1214e07643d.wav',
 '/home/shubrah/bird2/INSECT_2/c94efb21-04a0-4f48-9d77-7b078b66ddd0.wav',
 '/home/shubrah/bird2/INSECT_2/70b1f0e3-cece-4b28-a8ec-67135c39c258.wav',
 '/home/shubrah/bird2/INSECT_2/baea4bb4-7ca6-4017-a59f-6f075dd666d8.wav',
 '/home/shubrah/bird2/GRJF_s/64a831b4-9871-4ed5-a8d1-e7fc5781ae3c.wav',
 '/home/shubrah/bird2/GRJF_s/f741d662-09ea-4afb-827a-b24af90c22fa.wav',
 '/home/shubrah/bird2/INSECT_4/00bb76f1-f987-4ef0-9898-cb74210bc1b4.wav',
 '/home/shubrah/bird2/GRJF_2/8ca22cb3-aff6-4bda-b42a-0b3c2472bffa.wav',
 '/home/shubrah/bird2/GRJF_2/adb41c59-f97f-413a-be1e-3ad1bd50b7cd.wav',
 '/home/shubrah/bird2/ORHT_1/139bb21a-837f-4d85-99ec-edca79967c6c.wav',
 '/home/shubrah/bird2/ORHT_1/feccfd91-a78d-4ed1-809d-85db67bb6df5.wav',
 '/home/shubrah/bird2/INSB_s/4946e3fa-c0a8-47de-9ce7-5bbc70eaa396.wav',
 '/home/shubrah/bird2/INSB_s/e1ff2f52-3fc6-46ec-b827-07049e73b5bf.wav',
 '/home/shubrah/bird2/INSB_s/e2b2555a-0b5e-4b93-ad84-02e2fb4fabcf.wav',
 '/home/shubrah/bird2/INSB_s/0c7061c7-3326-4ee1-94fc-72357fe52da1.wav',
 '/home/shubrah/bird2/UNID_1/f606c4ef-afe7-4065-ab10-56312b04a81f.wav',
 '/home/shubrah/bird2/BULBUL_S/1843254a-62ec-4d99-aa24-0a030ef8e744.wav',
 '/home/shubrah/bird2/INSB_S/cc433a5e-371c-4b05-a6f9-c1755371874c.wav',
 '/home/shubrah/bird2/PUTB_1/d66ddd43-7ccf-4927-b1e8-348987cd664e.wav',
 '/home/shubrah/bird2/AFBB_1/da121bfa-6529-49d3-b0b3-0530543b440a.wav',
 '/home/shubrah/bird2/AFBB_1/72ab3f27-2361-4fa3-966c-81e1f6b6666f.wav',
 '/home/shubrah/bird2/AFBB_1/b4fc29d7-b2ae-4117-ace7-c57c949301da.wav',
 '/home/shubrah/bird2/AFBB_1/e155a73c-9831-4402-b809-d23f0d4b767d.wav',
 '/home/shubrah/bird2/AFBB_1/ecfab4a9-89ee-4c8a-a199-a07e1c9039a2.wav',
 '/home/shubrah/bird2/AFBB_1/29051a60-a242-4c22-8508-0ad4b712f6b6.wav',
 '/home/shubrah/bird2/AFBB_1/8c49021e-caca-423e-82b8-5d3eae267b90.wav',
 '/home/shubrah/bird2/AFBB_1/b825ab79-c183-407e-bd0e-a3956078b8cd.wav',
 '/home/shubrah/bird2/AFBB_1/84ef8caa-93e9-4477-8a44-f9b6f4be37f1.wav',
 '/home/shubrah/bird2/AFBB_1/5cf21e9c-e8e7-4ad5-9069-0c854a92d810.wav',
 '/home/shubrah/bird2/AFBB_1/090332a6-72d5-4882-a399-e580d44c2f41.wav',
 '/home/shubrah/bird2/AFBB_1/3e11372d-bf1f-4a54-adde-bc85c08ae6cc.wav',
 '/home/shubrah/bird2/AFBB_1/70388a91-9ad6-42d1-b83a-f3e857ae928a.wav',
 '/home/shubrah/bird2/AFBB_1/adf32a64-309b-4e97-b7dc-8de715704980.wav',
 '/home/shubrah/bird2/AFBB_1/111aee61-d0f9-425e-a842-4f4d65f6b09f.wav',
 '/home/shubrah/bird2/AFBB_1/3033fcb1-2f66-41e2-85bd-3c224cb66e1c.wav',
 '/home/shubrah/bird2/AFBB_1/bdc33dad-0121-4f27-8784-4958db0ccc31.wav',
 '/home/shubrah/bird2/AFBB_1/24ed927d-6f82-4310-8fa8-b0b6eff26031.wav',
 '/home/shubrah/bird2/AFBB_1/5ed8e91b-005b-4763-a06d-9e236b6048c3.wav',
 '/home/shubrah/bird2/AFBB_1/e18b0b5c-3f31-4666-8443-e39060b0a170.wav',
 '/home/shubrah/bird2/AFBB_1/b8ce351b-15bd-4713-ada0-458bf5f5f62d.wav',
 '/home/shubrah/bird2/AFBB_1/ed7583ac-1f31-470e-a260-75988e1412c6.wav',
 '/home/shubrah/bird2/AFBB_1/217cb577-c572-4cff-8752-807f89358ae1.wav',
 '/home/shubrah/bird2/AFBB_1/657f5aa6-6f87-48de-b718-141c9ff49cd1.wav',
 '/home/shubrah/bird2/AFBB_1/cfcb53b3-fee4-497a-9606-bb1039cf2bce.wav',
 '/home/shubrah/bird2/AFBB_1/82d52db9-fd40-4b3e-8882-e282b1739113.wav',
 '/home/shubrah/bird2/AFBB_1/c1c9e4da-8da4-4ff1-b02f-4ceb0b619b9a.wav',
 '/home/shubrah/bird2/AFBB_1/38a67959-e9f2-405d-a50f-33e06a91d1e2.wav',
 '/home/shubrah/bird2/AFBB_1/14b5a600-97f5-4bb8-a923-c8b622975929.wav',
 '/home/shubrah/bird2/AFBB_1/b88e4dbe-28ba-4557-952f-f1e0c14fbec0.wav',
 '/home/shubrah/bird2/GRCO_7/da1238c8-77fe-4d17-84ce-47e8c54897b4.wav',
 '/home/shubrah/bird2/INSECT_1/8c40629f-4a41-4689-a90d-60b4957536bf.wav',
 '/home/shubrah/bird2/INSECT_1/8bf46382-6aca-4901-914b-699d92750e43.wav',
 '/home/shubrah/bird2/INSECT_1/f20e5359-d56b-4078-82d1-13fbdbb5cd62.wav',
 '/home/shubrah/bird2/GRCO_2/a869748c-ba15-4726-920b-f8f94af0305e.wav',
 '/home/shubrah/bird2/UNID_s/c7ff8f4f-db4d-4a5f-ad35-31a988ca4edc.wav',
 '/home/shubrah/bird2/GRCO_s/b6a34d0a-d1d6-43d2-bda5-0fc999fb75ad.wav',
 '/home/shubrah/bird2/GRCO_s/2b660ba3-5b80-405a-8a96-4fe95a56693f.wav',
 '/home/shubrah/bird2/GRCO_s/94e75f9f-e44e-4f64-8348-a9b952d6130b.wav',
 '/home/shubrah/bird2/FLTB_2/4f2b072c-1c1e-4a70-89b5-e7e3c2ba09ca.wav',
 '/home/shubrah/bird2/AFBB_2/1f75d5d6-dcb0-44e7-aac3-c694a13f4c77.wav',
 '/home/shubrah/bird2/AFBB_2/d9f888ed-d860-416e-941b-6d5bbb56cd95.wav',
 '/home/shubrah/bird2/AFBB_2/4217eca0-8137-4bb6-9d87-25b8b7833012.wav',
 '/home/shubrah/bird2/GRJF_4/ce7513c9-fdab-4c40-8256-25a7153b2282.wav',
 '/home/shubrah/bird2/INSECT_7/02be15c5-efd1-40a4-ab3c-919f3cc0735f.wav',
 '/home/shubrah/bird2/FLTB_s/dd4c1037-0024-4777-8244-bdf80e4919f0.wav',
 '/home/shubrah/bird2/INSECT_5/6a9f360e-4f75-4433-ab15-0953d2a451ee.wav',
 '/home/shubrah/bird2/LABC_3/157b846a-8261-4047-81a0-9ef06e828145.wav',
 '/home/shubrah/bird2/PUTB_2/ed20e55f-face-46ec-8492-a32c70901a21.wav',
 '/home/shubrah/bird2/PUTB_2/46ad8d85-e5c9-401c-8608-d1d20be39efd.wav',
 '/home/shubrah/bird2/PUTB_2/eedfc084-5f0f-4328-b791-1a6fdb72757e.wav',
 '/home/shubrah/bird2/PUTB_2/58ceabca-aad2-4378-965c-9bdf69bbd2ed.wav',
 '/home/shubrah/bird2/GRJF_S/e975c764-ea62-4135-b567-692af1dcfb4a.wav',
 '/home/shubrah/bird2/ORHT_s/3ccc33d8-38a1-4daa-a295-61b20a75b186.wav',
 '/home/shubrah/bird2/ORHT_s/44b5e813-4932-4953-908b-a37ebdfe6e02.wav',
 '/home/shubrah/bird2/INSECT_3/83890764-5936-4292-87ee-3a3eae064f21.wav',
 '/home/shubrah/bird2/INSECT_3/186fd094-f741-4714-9496-71f894e435fa.wav',
 '/home/shubrah/bird2/INSECT_3/9fe2e999-bd1f-4be6-85b9-1c42ab002f5f.wav',
 '/home/shubrah/bird2/INSECT_s/f90a0921-a9a1-4c54-b513-95c3c00630a1.wav',
 '/home/shubrah/bird2/INSECT_s/2e4a3e29-0a3f-48d1-b2c8-04a90cda5ab4.wav',
 '/home/shubrah/bird2/INSECT_s/bcfb91f6-1721-4c29-8ca6-deabb2acf3d3.wav',
 '/home/shubrah/bird2/INSECT_s/f823eb2a-3613-45e9-b7f8-5f500a314e0a.wav']


extract3=['/home/shubrah/bird1/PUTB_s/f38d33d5-4a15-45d4-9b4c-e92e8a646e4a.wav',
 '/home/shubrah/bird1/PUSB_1/6d5cda23-b995-4877-af4c-1efc357ca0c9.wav',
 '/home/shubrah/bird1/EMDO_s/779b280e-19ad-46c8-bc9b-357672f98b0d.wav',
 '/home/shubrah/bird1/PUSB_25/bb0d0c45-5a55-4881-9b67-bb1c21d85c2e.wav',
 '/home/shubrah/bird1/EMDO_7/c2bbb28d-ccc3-4209-bc49-f2ac2619bd7a.wav',
 '/home/shubrah/bird1/REVB_1/b76204c2-137f-4fd8-8531-9789b939a63c.wav',
 '/home/shubrah/bird1/EMDO_4/ff2ce6ca-de4e-4840-af51-59f2617d2084.wav',
 '/home/shubrah/bird1/BNMF_1/e4141cde-84ff-4f68-a9da-254e65341ba0.wav',
 '/home/shubrah/bird1/RWBU_8/21e0f607-0c30-4cc2-b190-4d88d05fec4d.wav',
 '/home/shubrah/bird1/REWB_2/624306c5-d0ad-43ea-b32b-c17dba4f3dac.wav',
 '/home/shubrah/bird1/REVB_s/6e93f90e-749c-4906-9265-ceeb515024a2.wav',
 '/home/shubrah/bird1/INSB_s/c1f40e8e-456e-466b-829c-cc2f27f5e45e.wav',
 '/home/shubrah/bird1/UNID_1/eb416975-9500-4c42-9644-a71265daa167.wav',
 '/home/shubrah/bird1/UNID_1/5cc5bc5e-ce4d-4583-8f94-3a2816e0a993.wav',
 '/home/shubrah/bird1/UNID_1/68453067-71a4-4341-9318-6e6aa67473e3.wav',
 '/home/shubrah/bird1/BNMF_s/2f0c3117-bac7-47b9-82bc-bdc777b5c680.wav',
 '/home/shubrah/bird1/BNMF_s/8e24a867-5da5-4154-8907-0ea940664136.wav',
 '/home/shubrah/bird1/BNMF_s/b9ca08f2-f621-44b4-a004-d9270172a9b5.wav',
 '/home/shubrah/bird1/BNMF_s/813dafcc-d162-4784-9a5b-0e457e91cf19.wav',
 '/home/shubrah/bird1/BNMF_s/6d9e0da8-3644-4292-a68b-4c364c54c983.wav',
 '/home/shubrah/bird1/INSB_S/ad3e93c0-84b3-4be8-ab0c-db48225a7424.wav',
 '/home/shubrah/bird1/REWB_8/11697c4e-4452-4dd1-afae-8d65b07f6c60.wav',
 '/home/shubrah/bird1/INSECT_S/39d8a8a3-2517-4da1-9ad5-136329ba10d2.wav',
 '/home/shubrah/bird1/INSECT_1/a23259e8-787c-4227-ae73-371c97e9d438.wav',
 '/home/shubrah/bird1/INSECT_1/0be6eaa5-3f51-4fe0-a29f-114a7090890b.wav',
 '/home/shubrah/bird1/REVB_5/b593d552-60bb-4d02-9879-33a7c0d60058.wav',
 '/home/shubrah/bird1/RWWB_1/b09e5e8c-1f64-4590-8399-de3796551a09.wav',
 '/home/shubrah/bird1/REWB_s/9060d0cc-44b1-4e32-a0cc-8a13b7db961f.wav',
 '/home/shubrah/bird1/BNMF_S/d827a52d-6320-4445-8713-984743b04979.wav',
 '/home/shubrah/bird1/BNMF_S/ef791e7f-1d6d-4a16-a875-ddaceae0487a.wav',
 '/home/shubrah/bird1/GRTD_S/5ce94ac1-5837-4f81-92be-a13dbd4afc86.wav',
 '/home/shubrah/bird1/FROG_s/4853a23d-1773-4149-b209-6c9b78f4ccaf.wav',
 '/home/shubrah/bird1/FROG_s/e8fea68a-8f92-48d5-b65e-302a04d73813.wav',
 '/home/shubrah/bird1/REVB_2/d73afe9f-a7b4-459a-a697-732fc40b09d7.wav',
 '/home/shubrah/bird1/REVB_2/cddde6a5-27f4-4f6c-af2c-24203ecafa34.wav',
 '/home/shubrah/bird1/REWB_4/0ca639aa-2290-4b16-a48f-19eb5bee0571.wav',
 '/home/shubrah/bird1/INSECT_s/0553ae3c-14f5-48c7-90c4-94e396d50a3b.wav',
 '/home/shubrah/bird1/INSECT_s/c000d372-345c-417f-8399-1e4286a02bbe.wav',
 '/home/shubrah/bird1/INSECT_s/27d3fa12-79a8-46ef-a4c9-02d4bfa2cd29.wav']


for i in range(len(extract1)):
    create_spectrogram_test(extract1[i],extract1[i].split('/')[5].split('.')[0])
for i in range(len(extract3)):
    create_spectrogram_test(extract3[i],extract3[i].split('/')[5].split('.')[0])
testdf=create_df(extract1,'test')    

folder=extract3
some='test'
folder1=[]
for i in range(len(folder)):
    folder1.append([folder[i],folder[i].split('/')[4],'/home/shubrah/ml/pad/'+some+'/'+folder[i].split('/')[5].split('.')[0]+'.png'])
    a=pd.DataFrame(folder1,columns=['location','class','filename'])
    
testdf.append(a)

    
test_datagen=ImageDataGenerator(rescale=1./255.)


test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="/home/shubrah/ml/pad/test/",
    x_col='filename',
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


test_generator.reset()
pred=model.predict_generator(test_generator,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

#Fetch labels from train gen for testing
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions[0:10])
print(testdf.head(10))

acc=0
for i in range(len(testdf)):
    if predictions[i] == testdf["class"][i]:
        acc+=1
print(acc/len(testdf))
