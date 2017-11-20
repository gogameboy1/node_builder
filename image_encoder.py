import numpy as np
import glob
from scipy import misc
from sklearn.metrics.pairwise import cosine_similarity
import pdb

from keras.applications import vgg19
from keras.applications.vgg19 import preprocess_input
from keras.layers import Flatten
from keras.models import Model
from keras.preprocessing import image


PIC_PATH = 'penguin_topology_map/data/real/test1/topologyDB/image/'
TEST_PIC_PATH = ''


def load_data():

    image_list = []

    for image_file in glob.glob(TEST_PIC_PATH + '*.png'):
        # image = misc.imread(image_file)
        # image = misc.imresize(image, size=(224,224))
        # image = np.expand_dim(image, axis=0)
        # image_list.append(image)
        img = image.load_img(image_file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        image_list.append(img)

    return image_list


print ("loading model ...")
model = vgg19.VGG19(include_top=True,
                    weights='imagenet')

model.save("vgg19_pretrained.h5")


print ("loading data ...")
image_list = load_data()


print ("building image encoder ...")
layer_name = 'fc2'

image_encoder = Model(inputs=model.input,
                      outputs=model.get_layer(layer_name).output)

# save image vector list
image_vec_list = []

for image in image_list:

    image_vector = image_encoder.predict(image)

    image_vec_list.append(image_vector)

with open('image_vector.csv', 'w') as f:

    for index, vec in enumerate(image_vec_list):
        print ('image ' + str(index) + ',' + str(vec), file=f)


vector_1, vector_2, vector_3 = image_vec_list
pdb.set_trace()

print ("vector 1 and 1 : " + str(cosine_similarity(vector_1, vector_1)))

print ("vector 1 and 2 : " + str(cosine_similarity(vector_1, vector_2)))

print ("vector 1 and 3 : " + str(cosine_similarity(vector_1, vector_3)))

print ("vector 2 and 3 : " + str(cosine_similarity(vector_2, vector_3)))



