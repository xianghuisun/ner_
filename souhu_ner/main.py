from data_process import *

dev_source_data="./tmp/dev.txt"
dev_target_data="./tmp/dev-lable.txt"
source_data="./tmp/source.txt"
target_data="./tmp/target.txt"
test_source_data="./tmp/test1.txt"
test_target_data="./tmp/test_tgt.txt"



train_batch_size=64
epochs=15
train_file_path=(source_data,target_data)
dev_file_path=(dev_source_data,dev_target_data)
test_file_path=(test_source_data,test_target_data)

train_dataset=make_dataset(train_file_path,batch_size_=train_batch_size,epochs=epochs)
dev_dataset=make_dataset(dev_file_path,batch_size_=None,epochs=1,shuffle=False,mode="dev")
test_dataset=make_dataset(test_file_path,batch_size_=None,epochs=1,shuffle=False,mode="test")


with open("./parameter.pkl","rb") as f:
	word2id,label2id,max_seq_length=pickle.load(f)
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
model=build_model(fc_dim=fc_dim,batch_size_=train_batch_size,bilstm_layers=bilstm_layers,embedding_dim=embedding_dim)
def loss_fn(real_data,target_):
    mask=tf.math.logical_not(tf.math.equal(real_data,word2id["<pad>"]))
    mask=tf.cast(mask,dtype=tf.float32)
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
    loss=loss_object(real_data,target_)
    loss=loss*mask
    return tf.reduce_mean(loss)

model.compile(loss=loss_fn,optimizer=tf.keras.optimizers.Adam())

model.fit(train_dataset,epochs=epochs,verbose=2,validation_data=dev_dataset)

tf.saved_model.save(obj=model,export_dir="./simple_model")
print("*"*200)

def evaluate(test_file_path):
    data_list,label_list=read_data(source_data=test_file_path[0],target_data=test_file_path[1])
    with open("./parameter.pkl","rb") as f:
        word2id,label2id,max_seq_length=pickle.load(f)
    id_data_list,id_label_list=sentence_to_id(data_list=data_list,label_list=label_list,word2id=word2id,
                                             label2id=label2id)
    true_length_list=[]
    for each_sentence in id_data_list:
        true_length_list.append(len(each_sentence))
    pad_sentence,pad_label=pad_fn(id_data_list,id_label_list,max_seq_length,word2id)
    assert type(pad_sentence)==type(pad_label)==np.ndarray
    new_model=tf.saved_model.load(export_dir="./simple_model/")
    inferences=new_model.signatures["serving_default"]
    inferences.structured_outputs
    results=inferences(tf.constant(pad_sentence))
    predict=results["output_1"]
    predict=tf.argmax(predict.numpy(),axis=-1)
    assert predict.shape==pad_label.shape
    total_=0
    guess=0
    for each_predict_list,each_true_list,actual_length in zip(predict,pad_label,true_length_list):
        each_predict_list=each_predict_list[:actual_length]
        each_true_list=each_true_list[:actual_length]
        for each_predict,each_true in zip(each_predict_list,each_true_list):
            each_predict=each_predict.numpy()
            if each_predict==each_true:
                guess+=1
            total_+=1
    print("correctly predict number / total number is ",guess/total_)
    return new_model

new_model=evaluate(test_file_path)

print_sentence="武汉长江大桥是中国湖北省武汉市连接汉阳区与武昌区的过江通道"
sentence_id=[word2id.get(word,word2id["<unk>"]) for word in print_sentence]
input_what=tf.constant([sentence_id])
input_tensor=tf.keras.preprocessing.sequence.pad_sequences(input_what,value=word2id["<pad>"],padding="post",maxlen=max_seq_length)
output=new_model(input_tensor)
output=tf.squeeze(output)
output=output[:len(print_sentence)]
output=tf.argmax(output,axis=-1)
output.shape==(len(print_sentence),)
id2label={k:v for v,k in label2id.items()}

for predict_id,word in zip(output,print_sentence):
	print(word,"  ",id2label[predict_id.numpy()])
	
	

