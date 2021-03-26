import numpy as np
from torch.utils.data import Dataset
from fasttext import load_model
import os
import numpy as np
import torch
import h5py
import pickle
import re
import utils
from collections import Counter
from transformers import BertTokenizer

SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features

def word_tokenize(word):
    word = word.lower()
    word = word.replace(",", "").replace("?", "").replace("'s", " 's")
    return word.strip()

# 处理句子
def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]):
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

# vocabution
class Dictionary():
    def __init__(self):
        super(Dictionary, self).__init__()
    
        self.id_word = []
        self.word_id = {}
        with open('../data/vocabs/vocabulary_100k.txt','r') as file:
            f = file.readlines()
            for i in range(len(f)):
                self.id_word.append(word_tokenize(f[i]))
        for i in range(len(self.id_word)):
            self.word_id[self.id_word[i]] = i

    def __len__(self):
        return len(self.id_word)
    
    def get_index_by_word(self, word):
        return self.word_id[word]
    
    @property
    def padding_index(self):
        return len(self.id_word)
    
    def get_question_sequence(self, question_tokens, max_length = 14):
        
        question_sequence = []
        for k in question_tokens:
            question_sequence.append(self.word_id.get(k, self.padding_index))
        question_sequence = question_sequence[:max_length]
        
        if len(question_sequence) < max_length:
            padding = [self.padding_index] * (max_length - len(question_sequence))
            question_sequence += padding
        
        utils.assert_eq(len(question_sequence), max_length)
        return question_sequence

    def create_glove_embedding_init(self, glove_file = '../pythia/.vector_cache/glove.6B.300d.txt', pre=False, pre_dir=None):
        if pre:
            weights = np.load(open(pre_dir, 'rb'), allow_pickle=True)
        else:
            word2emb = {}
            with open(glove_file, 'r') as f:
                entries = f.readlines()
            emb_dim = len(entries[0].split(' ')) - 1

            print('embedding dim is %d' % emb_dim)
            weights = np.zeros((len(self.id_word), emb_dim), dtype=np.float32)

            for entry in entries:
                vals = entry.split(' ')
                word = vals[0]
                vals = list(map(float, vals[1:]))
                word2emb[word] = np.array(vals)  # word embedding

            count = 0
            for idx, word in enumerate(self.id_word):
                if word not in word2emb:
                    updates = 0
                    for w in word.split(' '):
                        if w not in word2emb:
                            continue
                        weights[idx] += word2emb[w]
                        updates += 1
                    if updates == 0:
                        count+= 1
                    continue
                weights[idx] = word2emb[word]
            print("%d 不在glove中"%count)
            np.save('../data/vocabs/embedding_weight.npy', weights)
            print("save success!")
        self.embedding_dim = weights.shape[1]
        return weights

class Answer(object):
    def __init__(self, answer_dir = '../data/vocabs/answer_index.pkl', pre = False, copy = True):
        super(Answer,self).__init__()
        self.answer_dir = answer_dir
        if pre:
            self.index2answer, self.answer2index =  pickle.load(open('../data/vocabs/answer_index.pkl','rb'))
            
        else:
            self.index2answer = []
            self.answer2index = {}
            self.answer_vocab('../data/vocabs/textvqa_more_than_8_unsorted.txt')
        
        self.max_num = 50
        
        self.candidate_answer_num = self.max_num + (self.length if copy else 0)
        self.noexisting_answer = 0
    
    ## 加载答案文档
    def answer_vocab(self, answer_vocab_dir, ):
        with open(answer_vocab_dir, 'r') as file:
            answers = file.readlines()
            for i in range(len(answers)):
                w = word_tokenize(answers[i])
                self.index2answer.append(w)
                self.answer2index[w] = i
        pickle.dump((self.index2answer, self.answer2index), open(self.answer_dir,'wb'))
    @property
    def length(self):
        return len(self.index2answer)
    
    '''
    answer source：
    no existing : 00 = 0
    classification : 01 = 1
    ocr : 10 = 2
    ocr and classification : 11 = 3
    '''
    
    def process_question_answer(self, answers, tokens = None):
        a = {}
        for i in range(len(answers)):
            w = word_tokenize(answers[i])
            a[w] = a.get(w,0) + 1
#         print(a)
        answer_source = {}
        answer_scores = torch.zeros(self.candidate_answer_num ,dtype = torch.float)
        for key in a:
            a[key] = min(1,a[key]/3)
            index = self.answer2index.get(key, None)
            if index != None:
                answer_scores[index] = a[key] 
                answer_source[key] = answer_source.get(key,0) + 1
        
        if tokens != None:
            token_scores = torch.zeros(self.max_num, dtype=torch.float)
            for i, t in enumerate(tokens[:self.max_num]):
                if t in a:
                    token_scores[i] = a[t]
                    answer_source[t] = answer_source.get(t, 0) + 2
        
        answer_scores[-self.max_num:] = token_scores
        if answer_scores.sum() == 0:
#             answer_scores[0] = 1   # 去掉unknown 这个标签，保证类平衡
            self.noexisting_answer += 1
            
        return answer_scores, answer_source
    
    def get_answer(self, index, token=None):
        if index>self.length:
            return token[index-self.length]
        else:
            return self.index2answer[index]

def get_ocr_bb(ocr_tokens, ocr_info):
    ocr_bb = []
    for ot in ocr_tokens:
        for oi in ocr_info:
            if oi["word"]== ot:
                ocr_bb.append(get_bb(oi["bounding_box"]))
    return ocr_bb

def get_bb(bb):
    return [bb['top_left_x'], bb['top_left_y'], bb['top_left_x']+bb["width"], bb['top_left_y']++bb["height"]]

# 整理数据集的每一项
def _load_dataset(name, dataroot = '../data/imdb/textvqa_0.5/'):
    """Load entries
    """
    question_path = os.path.join( dataroot, 'imdb_textvqa_%s.npy' % name)
    question_data = np.load(open(question_path, "rb"), allow_pickle=True)[1:]
    
    # delete the sample with empty ocr
    l = len(question_data)
    print("Total %d %s samples."%(l,name))
#     if name == "train":
#         i = 0
#         while i < l:
#             if question_data[i]["ocr_info"]==[]:
#                 question_data = np.delete(question_data,i)
#                 l = len(question_data)
#             else:
#                 i += 1
    print("Use %d %s samples."%(len(question_data), name))
    
    questions = sorted(question_data, key=lambda x: x['question_id'])
    entries = []
    for question in questions:
        tokens = question['ocr_tokens']
        ocr_bb = get_ocr_bb(tokens, question['ocr_info'])
        for i,w in enumerate(tokens):
            tokens[i] = word_tokenize(w)
        entries.append({
            'question_id' : question['question_id'],
            'image_id'    : question['image_id'],
            'question'    : question['question'],
            'ocr_tokens'  : tokens,
            'ocr_bb'  :     ocr_bb,
            'answer'      : (question['valid_answers'] if name!='test' else None)
        })
    
    return entries

# 数据集
class TextVQA(Dataset):
    def __init__(self, dataset, dictionary):
        
        super(TextVQA, self).__init__()
        assert dataset in ['train', 'val','test']
        self.name = dataset
        self.dictionary = dictionary
        # load image feature file
        
        image_feature_dir = '../data/open_images/detectron_fix_100/fc6/'
        if dataset == "val" or dataset == "train":
            h5_path = image_feature_dir + 'trainval.hdf5'
            imageid2index =  image_feature_dir+'trainval_imageid2index.pkl'
        else:
            h5_path = image_feature_dir + 'test.hdf5'
            imageid2index =  image_feature_dir+'test_imageid2index.pkl'
        
        self.img2index = pickle.load(open(imageid2index,'rb'))
        
        with h5py.File(h5_path, 'r') as hf:
            self.image_features = np.array(hf.get('image_features'))
            self.image_features_spatials = np.array(hf.get('spatial_features'))
        
        # load context data
        data_dir = '../data/imdb/textvqa_0.5/'
        context_feature_file = data_dir + "context_embeddding.hdf5"
        context_imageid2index = data_dir + "context_imageid2index.pkl"
        imageid2contextnum = data_dir + "imageid2contextnum.pkl"
        
        self.c_imageid_i = pickle.load(open(context_imageid2index,'rb'))
        self.c_imageid_num = pickle.load(open(imageid2contextnum,'rb'))
        with h5py.File(context_feature_file, 'r') as hf:
            self.c_features = np.array(hf.get('context_embedding'))
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
        self.answer_process = Answer(pre = False)
        self.entries = _load_dataset(dataset)
        
#         self.tokenize()
        self.tensorize()
        print("no existing answer",self.answer_process.noexisting_answer)
        
    def tokenize(self):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            entry['q_token'] = self.dictionary.get_question_sequence(entry['question'])
        
    def tensorize(self):
        self.image_features = torch.from_numpy(self.image_features)
        self.image_features_spatials = torch.from_numpy(self.image_features_spatials)
        self.c_features = torch.from_numpy(self.c_features)
        
        for entry in self.entries:
            train_features = convert_sents_to_features([entry['question']] ,15 , self.tokenizer)
            input_ids = torch.tensor(train_features[0].input_ids, dtype=torch.long)
            input_mask = torch.tensor(train_features[0].input_mask, dtype=torch.long)
            segment_ids = torch.tensor(train_features[0].segment_ids, dtype=torch.long)
            question = [input_ids, input_mask, segment_ids]
            entry['q_token'] = question
            
            ocr_bb = entry['ocr_bb']
            if ocr_bb != []:
                entry['ocr_bb'] = torch.from_numpy(np.array(ocr_bb))
            
            answer = entry['answer']
            if None != answer:
                entry['answer_label'], entry["answer_source"] = self.answer_process.process_question_answer(answer, entry['ocr_tokens'])
            else :
                entry['answer_label'] = None
        
    def __getitem__(self, index):
        
        image_id = self.entries[index]['image_id']
        
        c_feature = self.c_features[self.c_imageid_i[image_id]: self.c_imageid_i[image_id]+ self.c_imageid_num[image_id]]
        c_feature = c_feature[:50]
        
        context_feature = torch.zeros((50, c_feature.size(1)), dtype = torch.float)
        context_feature[:c_feature.size(0)] = c_feature
        
        ocr_bb_feature = torch.zeros((50, 4), dtype = torch.float)
        ocr_bb = self.entries[index]['ocr_bb']
        if not isinstance(ocr_bb,list):
            ocr_bb_feature[:ocr_bb.size(0)] = ocr_bb[:50]
        
        img_feature = self.image_features[self.img2index[image_id]]
        bbox = self.image_features_spatials[self.img2index[image_id]][:,:4]
        
        question = self.entries[index]['q_token']
        question_id = self.entries[index]['question_id']
        answer = self.entries[index]['answer_label']
        tokens = self.entries[index]['ocr_tokens']
        
#         order_vectors = torch.eye(context_feature.size(0))
#         order_vectors[len(tokens):] = 0
        
        sample = {
            "question_id" : question_id,
            "input_ids" : question[0],
            "token_type_ids" : question[2],
            "attention_mask" : question[1],
            "img_feature" : img_feature,
            "bbox": bbox,
            "context_feature" : context_feature,
            "ocrbbox":  ocr_bb_feature,
            "tokens" : len(tokens),
        }
        
        if answer is not None:
            sample["answer"] = answer
        return sample
    
    def __len__(self):
        return len(self.entries)
    
    def get_tokens_by_qId(self, q_Id):
        for i in range(len(self.entries)):
            if self.entries[i]["question_id"] == q_Id:
                return self.entries[i]['ocr_tokens']
    
    def get_answer_by_qId(self, q_Id):
        if self.name=="test":
            return None, None
        for i in range(len(self.entries)):
            if self.entries[i]["question_id"] == q_Id:
                answers = self.entries[i]["answer"]
                a = {}
                for i in range(len(answers)):
                    w = word_tokenize(answers[i])
                    a[w] = a.get(w,0) + 1
                    
                return a, self.entries[i]["answer_source"]