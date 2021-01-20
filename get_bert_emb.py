from param import *
from models import Basic_Bert_Unit_model
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW

import numpy as np

import time



def entlist2emb(Model,entids,entid2data,cuda_num):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb

def cos_sim_mat_generate(emb1,emb2,bs = 128,cuda_num = 0):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2,dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2,dim=1)
    res_mat = batch_mat_mm(array_emb1,array_emb2.t(),cuda_num,bs=bs)
    return res_mat

def batch_mat_mm(mat1,mat2,cuda_num,bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].cuda(cuda_num)
        res = temp_div_mat_1.mm(mat2.cuda(cuda_num))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat

def batch_topk(mat,bs=128,topn = 50,largest = False,cuda_num = 0):
    #be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat = mat[i:min(i+bs,axis_0)].cuda(cuda_num)
        score_mat,index_mat =temp_div_mat.topk(topn,largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score,0)
    res_index = torch.cat(res_index,0)
    return res_score,res_index

def generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                entid2data1,entid2data2,index2entity1,index2entity2,
                                nearest_sample_num = NEAREST_SAMPLE_NUM, batch_size = CANDIDATE_GENERATOR_BATCH_SIZE):
    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        #langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0,len(train_ent1s),batch_size):
            temp_emb = entlist2emb(Model,train_ent1s[i:i+batch_size],entid2data1,CUDA_NUM).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0,len(for_candidate_ent2s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent2s[i:i+batch_size],entid2data2,CUDA_NUM).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        #language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0,len(train_ent2s),batch_size):
            temp_emb = entlist2emb(Model,train_ent2s[i:i+batch_size],entid2data2,CUDA_NUM).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0,len(for_candidate_ent1s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent1s[i:i+batch_size],entid2data1,CUDA_NUM).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        #cos sim
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1,for_candidate_emb1)
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2,for_candidate_emb2)
        torch.cuda.empty_cache()
        #topk index
        _,topk_index_1 = batch_topk(cos_sim_mat1,topn=nearest_sample_num,largest=True)
        topk_index_1 = topk_index_1.tolist()
        _,topk_index_2 = batch_topk(cos_sim_mat2,topn=nearest_sample_num,largest=True)
        topk_index_2 = topk_index_2.tolist()
        #get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)
    print("get candidate using time: {:.3f}".format(time.time()-start_time))
    torch.cuda.empty_cache()
    return candidate_dict


def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data1,entid2data2):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s,pe2s,ne1s,ne2s in Train_gene:
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,entid2data1,cuda_num=CUDA_NUM)
        pos_emb2 = entlist2emb(Model,pe2s,entid2data2,cuda_num=CUDA_NUM)
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,entid2data1,cuda_num=CUDA_NUM)
        neg_emb2 = entlist2emb(Model,ne2s,entid2data2,cuda_num=CUDA_NUM)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM) #pos_score < neg_score
        batch_loss = Criterion( pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        batch_loss.backward()
        Optimizer.step()

        all_loss += batch_loss.item() * batch_length
    all_using_time = time.time()-start_time
    return all_loss,all_using_time

def hit_res(index_mat):
    ent1_num,ent2_num = index_mat.shape
    topk_n = [0 for _ in range(ent2_num)]
    for i in range(ent1_num):
        for j in range(ent2_num):
            if index_mat[i][j].item() == i:
                for h in range(j,ent2_num):
                    topk_n[h]+=1
                break
    topk_n = [round(x/ent1_num,5) for x in topk_n]
    print("hit @ 1: {:.5f}    hit @10 : {:.5f}    ".format(topk_n[1 - 1],topk_n[10 - 1]),end="")
    if ent2_num >= 25:
        print("hit @ 25: {:.5f}    ".format(topk_n[25 - 1]),end="")
    if ent2_num >= 50:
        print("hit @ 50: {:.5f}    ".format(topk_n[50 - 1]),end="")
    print("")

def save(Model,model_save_path):
    print("Model save in: ", model_save_path)
    Model.eval()
    torch.save(Model.state_dict(),model_save_path)

def train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,entid2data1,entid2data2,model_save_path):
    print("start training...")
    for epoch in range(EPOCH_NUM+1):
        print("+++++++++++")
        print("Epoch: ",epoch)
        print("+++++++++++")
        #generate candidate_dict
        #(candidate_dict is used to generate negative example for train_ILL)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2
        print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
              .format(len(train_ent1s),len(train_ent2s),len(for_candidate_ent1s),len(for_candidate_ent2s)))
        candidate_dict = generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                                 entid2data1,entid2data2,Train_gene.index2entity1,Train_gene.index2entity2)
        Train_gene.train_index_gene(candidate_dict) #generate training data with candidate_dict

        #train
        epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data1,entid2data2)
        Optimizer.zero_grad()
        torch.cuda.empty_cache()
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        if epoch >= 0:
            if epoch == EPOCH_NUM:
                save(Model,model_save_path)
            # test(Model,train_ill,entid2data,TEST_BATCH_SIZE,context="EVAL IN TRAIN SET")
            test(Model, test_ill, entid2data1, entid2data2, TEST_BATCH_SIZE, context="EVAL IN TEST SET:")


def test(Model,ent_ill,entid2data1, entid2data2,batch_size,context = ""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data1,CUDA_NUM).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data2,CUDA_NUM).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb1,emb2,batch_size,cuda_num=CUDA_NUM)
        score,top_index = batch_topk(res_mat,batch_size,topn = TOPK,largest=True,cuda_num=CUDA_NUM)
        hit_res(top_index)
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")

class Batch_TrainData_Generator(object):
    def __init__(self,train_ill,ent_ids1,ent_ids2,index2entity1,index2entity2,batch_size,neg_num):
        self.ent_ill = train_ill
        self.ent_ids1 = ent_ids1
        self.ent_ids2 = ent_ids2
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.iter_count = 0
        self.index2entity1 = index2entity1
        self.index2entity2 = index2entity2
        print("In Batch_TrainData_Generator, train ill num: {}".format(len(self.ent_ill)))
        print("In Batch_TrainData_Generator, ent_ids1 num: {}".format(len(self.ent_ids1)))
        print("In Batch_TrainData_Generator, ent_ids2 num: {}".format(len(self.ent_ids2)))
        # print("In Batch_TrainData_Generator, keys of index2entity num: {}".format(len(self.index2entity)))

    def train_index_gene(self,candidate_dict):
        """
        generate training data (entity_index).
        """
        train_index = [] #training data
        candid_num = 999999
        for ent in candidate_dict:
            candid_num = min(candid_num,len(candidate_dict[ent]))
            candidate_dict[ent] = np.array(candidate_dict[ent])
        for pe1,pe2 in self.ent_ill:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    #e1
                    ne1 = candidate_dict[pe2][np.random.randint(candid_num)]
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2 = candidate_dict[pe1][np.random.randint(candid_num)]
                #same check
                if pe1!=ne1 or pe2!=ne2:
                    train_index.append([pe1,pe2,ne1,ne2])
        np.random.shuffle(train_index)
        self.train_index = train_index
        self.batch_num = int( np.ceil( len(self.train_index) * 1.0 / self.batch_size ) )

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1

            batch_data = self.train_index[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]

            pe1s = [pe1 for pe1,pe2,ne1,ne2 in batch_data]
            pe2s = [pe2 for pe1,pe2,ne1,ne2 in batch_data]
            ne1s = [ne1 for pe1,pe2,ne1,ne2 in batch_data]
            ne2s = [ne2 for pe1,pe2,ne1,ne2 in batch_data]

            return pe1s,pe2s,ne1s,ne2s

        else:
            del self.train_index
            self.iter_count = 0
            raise StopIteration()

if __name__ == '__main__':
    print("fine-tune embeddings start...")
    BERT_Model = Basic_Bert_Unit_model(bert_dim, ent_dim)
    BERT_Model.cuda(cuda_num)

    Criterion = nn.MarginRankingLoss(MARGIN,size_average=True)
    Optimizer = AdamW(BERT_Model.parameters(),lr=LEARNING_RATE)

    index2entity1 = process_pickle(index2entity1_path)
    index2entity2 = process_pickle(index2entity2_path)
    index2rel1 = process_pickle(index2rel1_path)
    index2rel2 = process_pickle(index2rel2_path)

    ent2data1 = process_pickle(ent2data1_path)
    ent2data2 = process_pickle(ent2data2_path)

    rel2data1 = process_pickle(rel2data1_path)
    rel2data2 = process_pickle(rel2data2_path)


    train_ey = process_pickle(train_ey_path)
    test_ey = process_pickle(test_ey_path)

    train_ry = process_pickle(train_ry_path)
    test_ry = process_pickle(test_ry_path)

    ey = process_pickle(ey_path)
    ry = process_pickle(ry_path)

    ent1 = [e1 for e1,e2 in ey]
    ent2 = [e2 for e1,e2 in ey]

    rel1 = [r1 for r1,r2 in ry]
    rel2 = [r2 for r1,r2 in ry]

    Ent_Train_gene = Batch_TrainData_Generator(train_ey, ent1, ent2, index2entity1, index2entity2, batch_size=TRAIN_BATCH_SIZE, neg_num=NEG_NUM)
    Rel_Train_gene = Batch_TrainData_Generator(train_ry, rel1, rel2, index2rel1, index2rel2, batch_size=TRAIN_BATCH_SIZE, neg_num=NEG_NUM)
    train(BERT_Model,Criterion,Optimizer,Ent_Train_gene,train_ey,test_ey,ent2data1,ent2data2,ENT_MODEL_SAVE_PATH)
    train(BERT_Model,Criterion,Optimizer,Rel_Train_gene,train_ry,test_ry,rel2data1,rel2data2,REL_MODEL_SAVE_PATH)

