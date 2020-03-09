import tensorflow as tf
import numpy as np
import pickle
with open("./word_label2id.pkl","rb") as f:
    word2id,label2id=pickle.load(f)

vocab_size=len(word2id)
num_classes=len(label2id)
class build_model(tf.keras.Model):
    def __init__(self,fc_dim,batch_size_,embedding_dim,bilstm_layers):
        super(build_model,self).__init__()
        self.batch_size=batch_size_
        self.fc_dim=fc_dim
        forward_layer1,backward_layer1,forward_layer2,backward_layer2=bilstm_layers
        self.Embedding_layer=tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,
                                                       batch_input_shape=[self.batch_size,max_seq_length])
        self.bilstm_layer1=tf.keras.layers.Bidirectional(layer=forward_layer1,backward_layer=backward_layer1,merge_mode="concat")
        self.bilstm_layer2=tf.keras.layers.Bidirectional(layer=forward_layer2,backward_layer=backward_layer2,merge_mode="concat")
        self.fc_layer=tf.keras.layers.Dense(units=fc_dim,activation="relu")
        self.output_layer=tf.keras.layers.Dense(units=num_classes)
        #(batch_size,max_seq_length,num_classes)
    def call(self,input_tensor):
        return self.output_layer(self.fc_layer(self.bilstm_layer2(self.bilstm_layer1(self.Embedding_layer(input_tensor)))))
max_seq_length=178
fc_dim=128
embedding_dim=100
forward_layer1=tf.keras.layers.LSTM(units=64,return_sequences=True,go_backwards=False,time_major=False)
backward_layer1=tf.keras.layers.LSTM(units=64,return_sequences=True,go_backwards=True,time_major=False)
forward_layer2=tf.keras.layers.LSTM(units=128,return_sequences=True,go_backwards=False,time_major=False)
backward_layer2=tf.keras.layers.LSTM(units=128,return_sequences=True,go_backwards=True,time_major=False)
bilstm_layers=(forward_layer1,backward_layer1,forward_layer2,backward_layer2)

evaluate_model=build_model(fc_dim=fc_dim,batch_size_=1,bilstm_layers=bilstm_layers,embedding_dim=embedding_dim)

def loss_fn(real_data,target_):
    mask=tf.math.logical_not(tf.math.equal(real_data,word2id["<pad>"]))
    mask=tf.cast(mask,dtype=tf.float32)
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
    loss=loss_object(real_data,target_)
    loss=loss*mask
    return tf.reduce_mean(loss)

evaluate_model.compile(loss=loss_fn,optimizer=tf.keras.optimizers.Adam())

evaluate_model.load_weights(filepath="./simple_model.ckpt")

id2label={k:v for v,k in label2id.items()}

test_sentence="武汉市是湖北省的省会，马云是阿里巴巴的创始人"
test_sentence=[word2id.get(word,word2id["<unk>"]) for word in test_sentence]
test_padding=tf.keras.preprocessing.sequence.pad_sequences([[test_sentence]],maxlen=178,padding="post",value=0)

result=evaluate_model(test_padding)
assert result.shape==(1,178,num_classes)
result=tf.squeeze(result,axis=0)
result=result[:len(test_sentence)]
predict=result.numpy()
assert len(predict)==len(test_sentence)
for word,tag_id in zip(test_sentence,predict):
    print(word,id2label[tag_id])
