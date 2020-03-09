import tensorflow as tf
import numpy as np

import pickle



def get_data(file_path):
    with open(file=file_path,mode='r',encoding="utf-8") as f:
        lines=f.readlines()
    all_sentence_list=[]
    all_label_list=[]
    sentence_list=[]
    label_list=[]
    for line in lines:
        if line=="\n" and len(sentence_list)!=0:
            all_sentence_list.append(sentence_list)
            all_label_list.append(label_list)
            sentence_list=[]
            label_list=[]
        line_split=line.strip().split()
        if len(line_split)!=2:
            continue
        assert len(line_split)==2
        word=line_split[0]
        label=line_split[1]
        sentence_list.append(word)
        label_list.append(label)
    return all_sentence_list,all_label_list

train_file="./ResumeNER/train.char.bmes"
test_file="./ResumeNER/test.char.bmes"
dev_file="./ResumeNER/dev.char.bmes"
#data1,target1=get_data(train_file)
#data2,target2=get_data(test_file)
#data3,target3=get_data(dev_file)
#data=data1+data2+data3
#target=target1+target2+target3
#
#print(len(data),len(target))

def get_word_label2id(sentence_list,label_list):
    word2id={}
    label2id={}
    all_words=[]
    label_set=set()
    word2id["<pad>"]=len(word2id)
    from collections import Counter
    for sentence,label_sentence in zip(sentence_list,label_list):
        for word in sentence:
            all_words.append(word)
        for label in label_sentence:
            label_set.add(label)
    label_=list(label_set)
    for label in label_:
        label2id[label]=len(label2id)
    counter=Counter(all_words)
    import operator
    sorted_list=sorted(counter.items(),key=operator.itemgetter(1),reverse=True)
    for word,freq in sorted_list:
        word2id[word]=len(word2id)
    word2id["<unk>"]=len(word2id)
    
    parameter=(word2id,label2id)
    with open("./word_label2id.pkl","wb") as f:
        pickle.dump(parameter,f)

#get_word_label2id(data,target)
#sentence_list,label_list=data,target
with open("./word_label2id.pkl","rb") as f:
    word2id,label2id=pickle.load(f)

#def sentence_to_id(sentence_list,label_list):
#    id_sentence_list=[]
#    id_label_list=[]
#    for sentence,label_sentence in zip(sentence_list,label_list):
#        assert len(sentence)==len(label_sentence)
#        id_sentence_list.append([word2id[word] for word in sentence])
#        id_label_list.append([label2id[label] for label in label_sentence])
#    return id_sentence_list,id_label_list
#id_sentence_list,id_label_list=sentence_to_id(sentence_list,label_list)
#max_seq_length=max(len(sentence) for sentence in id_sentence_list)
max_seq_length=178
def pad_function(id_sentence_list,id_label_list):
    assert len(id_sentence_list)==len(id_label_list)
    for id_sentence,id_label in zip(id_sentence_list,id_label_list):
        assert len(id_sentence)==len(id_label)
    pad_sentences=tf.keras.preprocessing.sequence.pad_sequences(id_sentence_list,padding="post",
                                                 value=word2id["<pad>"],maxlen=max_seq_length)
    pad_labels=tf.keras.preprocessing.sequence.pad_sequences(id_label_list,padding="post",
                                                            value=word2id["<pad>"],maxlen=max_seq_length)
    return pad_sentences,pad_labels
#pad_sentences,pad_labels=pad_function(id_sentence_list,id_label_list)

#dataset=tf.data.Dataset.from_tensor_slices((pad_sentences,pad_labels))
#train_batch_size=64
#dataset=dataset.shuffle(2000).batch(batch_size=train_batch_size,drop_remainder=True)
num_classes=len(label2id)
vocab_size=len(word2id)

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

fc_dim=128
embedding_dim=100
forward_layer1=tf.keras.layers.LSTM(units=64,return_sequences=True,go_backwards=False,time_major=False)
backward_layer1=tf.keras.layers.LSTM(units=64,return_sequences=True,go_backwards=True,time_major=False)
forward_layer2=tf.keras.layers.LSTM(units=128,return_sequences=True,go_backwards=False,time_major=False)
backward_layer2=tf.keras.layers.LSTM(units=128,return_sequences=True,go_backwards=True,time_major=False)
bilstm_layers=(forward_layer1,backward_layer1,forward_layer2,backward_layer2)

#model=build_model(fc_dim=fc_dim,batch_size_=train_batch_size,bilstm_layers=bilstm_layers,embedding_dim=embedding_dim)
def loss_fn(real_data,target_):
    mask=tf.math.logical_not(tf.math.equal(real_data,word2id["<pad>"]))
    mask=tf.cast(mask,dtype=tf.float32)
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
    loss=loss_object(real_data,target_)
    loss=loss*mask
    return tf.reduce_mean(loss)

#model.compile(loss=loss_fn,optimizer=tf.keras.optimizers.Adam())
#model.fit(dataset,epochs=30,verbose=2)
print("*"*200)
#model.save_weights(filepath="./simple_model.ckpt")
new_model=build_model(fc_dim=fc_dim,batch_size_=1,bilstm_layers=bilstm_layers,embedding_dim=embedding_dim)
new_model.compile(loss=loss_fn,optimizer=tf.keras.optimizers.Adam())

new_model.load_weights(filepath="./simple_model.ckpt")

print("*"*500)


test_sentence="武汉市是湖北省的省会，马云是阿里巴巴的创始人"
test_sentence_id=[word2id.get(word,word2id["<unk>"]) for word in test_sentence]
test_padding=tf.keras.preprocessing.sequence.pad_sequences([[test_sentence_id]],maxlen=178,padding="post",value=0)

result=new_model(test_padding)
assert result.shape==(1,178,num_classes)
result=tf.squeeze(result,axis=0)
result=result[:len(test_sentence)]
predict=tf.argmax(result,axis=-1)


for i,j in zip(predict.numpy(),test_sentence):
    print(id2label[i],": ",test_sentence[j])

