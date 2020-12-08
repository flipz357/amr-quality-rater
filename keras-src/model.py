from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate, Reshape, Embedding, Dense, Flatten, Subtract, Multiply
from keras.models import Model

class ModelBuilder():

    def __init__(self):
        return None

    def merge(self, inputs):
        m = Multiply()(inputs)
        s = Subtract()(inputs)
        return Concatenate()([m, s])
    
    def build_amr_quality_rater(self
            , vocab_count
            , amr_shape=(30,30)
            , dep_shape=(30,30)
            , n_main_output_neurons=1
            , n_aux_output_neurons=None):

        input_amr = Input(shape = (amr_shape[0] * amr_shape[1], )
                , dtype='int32'
                , name='token_input_amr')

        input_dep = Input(shape=(dep_shape[0] * dep_shape[1], )
                , dtype='int32'
                , name='token_input_dep')
    
        #(batch size, img height * img weidth, channels)
        amr_embedded = Embedding(input_dim=vocab_count+1
                , output_dim=128
                , input_length=amr_shape[0] * amr_shape[1]) (input_amr)

        dep_embedded = Embedding(input_dim=vocab_count+1
                , output_dim=128
                , input_length=dep_shape[0] * dep_shape[1]) (input_dep)
        
        #(batch size, img height, img weidth, channels)
        input_img_amr = Reshape((amr_shape[0], amr_shape[1], 128)) (amr_embedded)
        
        input_img_dep = Reshape((dep_shape[0] ,dep_shape[1], 128)) (dep_embedded)
        
        #(batch size, img height, img width, num kernels)
        amr_conv_1 = Conv2D(256, (3,3), padding="same", activation='relu') (input_img_amr)
        dep_conv_1 = Conv2D(256, (3,3), padding="same", activation='relu') (input_img_dep)
        
        #(batch size, img height/pool height, img width/pool width, num kernels)
        amr_pool_1 = MaxPooling2D((3,3), strides=(3,3), padding="valid") (amr_conv_1)
        dep_pool_1 = MaxPooling2D((3,3), strides=(3,3), padding="valid") (dep_conv_1)
        
        # similar to above
        amr_conv_2 = Conv2D(128, (5,5), padding="same", activation='relu') (amr_pool_1)
        dep_conv_2 = Conv2D(128, (5,5), padding="same", activation='relu') (dep_pool_1)
        
        # similar to above
        amr_pool = MaxPooling2D((5,5),strides=(5,5), padding="valid") (amr_conv_2)
        dep_pool = MaxPooling2D((5,5),strides=(5,5), padding="valid") (dep_conv_2)

        # vectorized global representations (batch size, dims) 
        flat_dep = Flatten()(dep_pool)
        flat_amr = Flatten()(amr_pool)

        # joint global representations (batch size, 2*dims) 
        shared = self.merge([flat_amr, flat_dep])
        
        # joint residual representations (batch size, 2*dims) 
        res_shared = Flatten()(
                MaxPooling2D(
                    (30,15),strides=(30,15), padding="valid") (self.merge([amr_conv_1, dep_conv_1])
                        )
                    )
        shared = Concatenate()([shared, res_shared]) 
        
        # joint final representation
        shared = Dense(256,activation="relu")(shared)
        
        if n_aux_output_neurons:
            aux_output = Dense(n_aux_output_neurons, activation="sigmoid")(shared) 
            shared = Concatenate()([shared, aux_output])
            output = Dense(n_main_output_neurons, activation="sigmoid")(shared) 
            regression_model = Model([input_amr, input_dep], [aux_output, output])
        else:
            output = Dense(n_main_output_neurons, activation="sigmoid")(shared) 
            regression_model = Model([input_amr, input_dep], output)
            
        print(regression_model.summary())
        return regression_model

    
