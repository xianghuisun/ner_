import tensorflow as tf
import numpy as np
from collections import Counter
import operator
import pickle

def read_data(source_data,target_data):
    with open(source_data,"r",encoding="utf-8") as f:
        data_lines=f.readlines()
    with open(target_data,"r",encoding="utf-8") as f:
        label_lines=f.readlines()
    data_list=[]
    label_list=[]
    assert len(data_lines)==len(label_lines)
    for data_sen,label_sen in zip(data_lines,label_lines):
        data_list.append(data_sen.strip().split())
        label_list.append(label_sen.strip().split())
    return data_list,label_list

def get_parameter(data_list,label_list):
    word2id={}
    label2id={}
    word2id["<pad>"]=len(word2id)
    all_words=[]
    label_set=set()
    for sentence_list,label_list in zip(data_list,label_list):
        assert len(sentence_list)==len(label_list)
        for word,label in zip(sentence_list,label_list):
            label_set.add(label)
            all_words.append(word)
    label_set=list(label_set)
    for label in label_set:
        label2id[label]=len(label2id)
    counter=Counter(all_words)
    sorted_list=sorted(counter.items(),key=operator.itemgetter(1),reverse=True)
    for word,freq in sorted_list:
        word2id[word]=len(word2id)
    word2id["<unk>"]=len(word2id)
    return word2id,label2id

def sentence_to_id(data_list,label_list,word2id,label2id):
    id_data_list=[]
    id_label_list=[]
    for sentence_list,label_list in zip(data_list,label_list):
        assert len(sentence_list)==len(label_list)
        id_data_list.append([word2id.get(word,word2id["<unk>"]) for word in sentence_list])
        id_label_list.append([label2id[label] for label in label_list])
    return id_data_list,id_label_list

def review_sentence(id_list,word2id):
    id2word={k:v for v,k in word2id.items()}
    review=[]
    for id_ in id_list:
        review.append(id2word.get(id_,"<unk>"))
    return review
def review_label(id_list,label2id):
    id2label={k:v for v,k in label2id.items()}
    review=[]
    for id_ in id_list:
        review.append(id2label.get(id_,"<unk>"))
    return review

def get_max_length(id_sentence_list):
    return max(len(id_sentence) for id_sentence in id_sentence_list)

def pad_fn(id_sentence_list,id_label_list,max_seq_length,word2id):
    pad_sentence=tf.keras.preprocessing.sequence.pad_sequences(id_sentence_list,
                                                               value=word2id["<pad>"],
                                                               padding="post",
                                                               maxlen=max_seq_length
                                                              )
    pad_label=tf.keras.preprocessing.sequence.pad_sequences(id_label_list,
                                                               value=word2id["<pad>"],
                                                               padding="post",
                                                               maxlen=max_seq_length
                                                              )
    return pad_sentence,pad_label

def make_dataset(file_path,batch_size_,epochs,shuffle=True,mode="train"):
    source_data,target_data=file_path
    data_list,label_list=read_data(source_data,target_data)
    if mode=="train":
        word2id,label2id=get_parameter(data_list,label_list)
        id_data_list,id_label_list=sentence_to_id(data_list,label_list,word2id,label2id)
        max_seq_length=get_max_length(id_data_list)
        parameter=(word2id,label2id,max_seq_length)
        with open("./parameter.pkl","wb") as f:
            pickle.dump(parameter,f)
    else:
        with open("./parameter.pkl","rb") as f:
            word2id,label2id,max_seq_length=pickle.load(f)
            id_data_list,id_label_list=sentence_to_id(data_list,label_list,word2id,label2id)
            
    pad_sentence,pad_label=pad_fn(id_data_list,id_label_list,max_seq_length,word2id)
    dataset=tf.data.Dataset.from_tensor_slices((pad_sentence,pad_label))
    print(dataset)
    if shuffle==True:
        shuffle_size=int(pad_sentence.shape[0]*3/5)
        dataset=dataset.repeat(epochs).shuffle(shuffle_size).batch(batch_size_)
        return dataset
    dataset=dataset.batch(pad_sentence.shape[0])
    return dataset





