from param import *
from utils import *
from models import Basic_Bert_Unit_model

from transformers import BertTokenizer
import numpy as np


def get_name(string):
    if r"resource/" in string:
        sub_string = string.split(r"resource/")[-1]
    elif r"property/" in string:
        sub_string = string.split(r"property/")[-1]
    else:
        sub_string = string.split(r"/")[-1]
    sub_string = sub_string.replace('_',' ')
    return sub_string

def read_id2object(file_path):
    id2object = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        print('loading a (id2object)file...  ' + file_path)
        for line in f:
            th = line.strip('\n').split('\t')
            id2object[int(th[0])] = th[1]
    return id2object

def ent2desTokens_generate(Tokenizer,des_dict_path,ent_list_1,ent_list_2,des_limit=des_limit):
    #ent_list_1/2 == two different language ent list
    print("load desription data from... :", des_dict_path)
    ori_des_dict = process_pickle(des_dict_path)
    ent2desTokens1 = dict()
    ent2desTokens2 = dict()
    ent_set_1 = set(ent_list_1)
    ent_set_2 = set(ent_list_2)
    for ent,ori_des in ori_des_dict.items():
        if ent in ent_set_1:
            string = ori_des
            encode_indexs = Tokenizer.encode(string)[:des_limit]
            ent2desTokens1[ent] = encode_indexs
        elif ent not in ent_set_2:
            string = ori_des
            encode_indexs = Tokenizer.encode(string)[:des_limit]
            ent2desTokens2[ent] = encode_indexs
        else:
            continue

    print("The num of entity with description 1:", len(ent2desTokens1.keys()))
    print("The num of entity with description 2:", len(ent2desTokens2.keys()))

    return ent2desTokens1, ent2desTokens2

def get_tokens_of_value(vaule_list,Tokenizer,max_length):
    #return tokens of attributeValue
    tokens_list = []
    for v in vaule_list:
        if 'http://zh.dbpedia.org/property/' in v:
            property_str = v[len('http://zh.dbpedia.org/property/'):]
        else:
            property_str = v[len('http://dbpedia.org/property/'):]
        token_ids = Tokenizer.encode(property_str, add_special_tokens=True,max_length=max_length)
        tokens_list.append(token_ids)
    return tokens_list

def obj2Tokens_gene(Tokenizer,obj2desTokens,index2obj,obj_name_max_length=des_limit):
    obj2tokenids = dict()
    for obj_id, obj_n in index2obj.items():
        if obj2desTokens!= None and obj_n in obj2desTokens:
            #if entity has description, use entity description
            token_ids = obj2desTokens[obj_n]
            obj2tokenids[obj_id] = token_ids
        else:
            #else, use entity name.
            ent_name = get_name(obj_n)
            token_ids = Tokenizer.encode(ent_name)[:obj_name_max_length]
            obj2tokenids[obj_id] = token_ids
    return obj2tokenids

def obj2bert_input(index2obj,Tokenizer,obj2token_ids,need_spec_tokens=True,des_max_length=des_max_length):
    obj2data = dict()
    pad_id = Tokenizer.pad_token_id

    for obj_id, obj_n in index2obj.items():
        obj2data[obj_id] = [[],[]]
        obj_token_id = obj2token_ids[obj_id]
        if need_spec_tokens:
            obj_token_ids = Tokenizer.build_inputs_with_special_tokens(obj_token_id)
        else:
            obj_token_ids = obj_token_id

        token_length = len(obj_token_ids)
        assert token_length <= des_max_length

        obj_token_ids = obj_token_ids + [pad_id] * max(0, des_max_length - token_length)

        obj_mask_ids = np.ones(np.array(obj_token_ids).shape)
        obj_mask_ids[np.array(obj_token_ids) == pad_id] = 0
        obj_mask_ids = obj_mask_ids.tolist()

        obj2data[obj_id][0] = obj_token_ids
        obj2data[obj_id][1] = obj_mask_ids
    return obj2data

def read_idtuple_file(file_path):
    print('loading a idtuple file...   ' + file_path)
    ret = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            th = line.strip('\n').split('\t')
            x = []
            for i in range(len(th)):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def read_spec_idtuple_file(file_path):
    print('loading a idtuple file...   ' + file_path)
    ret = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            th = line.strip('\n').split('\t')
            x = []
            for i in range(len(th[:-1])):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret

def get_edge_info(edge_triples):
    edge_index = []
    edge_rel = []
    for edge_tri in edge_triples:
        head_id = edge_tri[0]
        rel_id = edge_tri[1]
        tail_id = edge_tri[2]

        edge_index.append([head_id, tail_id])
        edge_rel.append(rel_id)

    return edge_index, edge_rel

def get_ent_aligns(ent_ills, s_ent_map, t_ent_map):
    ent_y = []
    for s_entid, t_entid in ent_ills:
        ent_y.append([s_ent_map[s_entid], t_ent_map[t_entid]])
    return ent_y

def get_obj_map(obj_tids):
    obj_map = {}
    for i, obj_id in enumerate(obj_tids):
        obj_map[obj_id] = i
    return obj_map

def index2obj_refine(index2obj, obj_map):
    refine_index2obj = {}
    for e_id, e_n in index2obj.items():
        refine_index2obj[obj_map[e_id]] = e_n
    return refine_index2obj

def get_ent_emb(ent_emb_dict, ent_map, entity2index):
    ent_embs = np.zeros([len(ent_emb_dict), ent_dim])
    for ent_n, ent_emb in ent_emb_dict.items():
        ent_id = entity2index[ent_n]
        ent_index = ent_map[ent_id]
        ent_embs[ent_index] = ent_emb
    return ent_embs

def get_rel_emb(rel_emb_dict, rel_map, relity2index):
    rel_embs = np.zeros([len(rel_emb_dict), rel_dim])
    for rel_n, rel_emb in rel_emb_dict.items():
        rel_id = relity2index[rel_n]
        rel_index = rel_map[rel_id]
        rel_embs[rel_index] = rel_emb
    return rel_embs

def get_tradic_edges(edge_index, edge_type, entity_cnt, ent_map, rel_map):
    edge_index = np.array(edge_index).T
    tradic_edge_index = []
    for i in range(edge_index.shape[1]):
        head = ent_map[edge_index[0, i]]
        tail = ent_map[edge_index[1, i]]
        relation = rel_map[edge_type[i]] + entity_cnt
        tradic_edge_index.append([head, relation])
        tradic_edge_index.append([head, tail])

    for i in range(edge_index.shape[1]):
        head = ent_map[edge_index[0, i]]
        tail = ent_map[edge_index[1, i]]
        relation = rel_map[edge_type[i]] + entity_cnt
        tradic_edge_index.append([tail, head])
        tradic_edge_index.append([tail, relation])
    # edge_type.append() 在忧郁要不要加上表示是关系还是实体的token

    for i in range(edge_index.shape[1]):
        head = ent_map[edge_index[0, i]]
        tail = ent_map[edge_index[1, i]]
        relation = rel_map[edge_type[i]] + entity_cnt
        tradic_edge_index.append([relation, tail])

    for i in range(edge_index.shape[1]):
        head = ent_map[edge_index[0, i]]
        tail = ent_map[edge_index[1, i]]
        relation = rel_map[edge_type[i]] + entity_cnt
        # relation = edge_type[i]
        tradic_edge_index.append([relation, head])

    return tradic_edge_index

def get_bert_entity_em(ent2data, BERT_Model, batch_size=bert_batch_size):
    ent_emb = []
    for eid in range(0, len(ent2data.keys()), batch_size):  # eid == [0,n)
        token_inputs = []
        mask_inputs = []
        for i in range(eid, min(eid + batch_size, len(ent2data.keys()))):
            token_input = ent2data[i][0]
            mask_input = ent2data[i][1]
            token_inputs.append(token_input)
            mask_inputs.append(mask_input)
        vec = BERT_Model(torch.LongTensor(token_inputs).cuda(cuda_num),torch.FloatTensor(mask_inputs).cuda(cuda_num))
        ent_emb.extend(vec.detach().cpu().tolist())
    return ent_emb

if __name__ == '__main__':

    # index2entity1 = read_id2object(ent_ids1_path)
    # index2entity2 = read_id2object(ent_ids2_path)
    #
    # entity_cnt1 = len(index2entity1)
    # entity_cnt2 = len(index2entity2)
    #
    # index2rel1 = read_id2object(rel_ids1_path)
    # index2rel2 = read_id2object(rel_ids2_path)
    #
    # ent_map1 = get_obj_map(index2entity1.keys())
    # ent_map2 = get_obj_map(index2entity2.keys())
    #
    # rel_map1 = get_obj_map(index2rel1.keys())
    # rel_map2 = get_obj_map(index2rel2.keys())
    #
    # index2entity1 = index2obj_refine(index2entity1, ent_map1)
    # index2entity2 = index2obj_refine(index2entity2, ent_map2)
    #
    # index2rel1 = index2obj_refine(index2rel1, rel_map1)
    # index2rel2 = index2obj_refine(index2rel2, rel_map2)
    #
    # generate_pickle(index2entity1_path, index2entity1)
    # generate_pickle(index2entity2_path, index2entity2)
    #
    # generate_pickle(index2rel1_path, index2rel1)
    # generate_pickle(index2rel2_path, index2rel2)
    #
    # entity2index1 = {e:idx for idx,e in index2entity1.items()}
    # entity2index2 = {e:idx for idx,e in index2entity2.items()}
    # rel2index1 = {e:idx for idx,e in index2rel1.items()}
    # rel2index2 = {e:idx for idx,e in index2rel2.items()}

    '''
    get tradic graph
    '''
    # rel_triples_1 = read_idtuple_file(triples1_path)
    # rel_triples_2 = read_idtuple_file(triples2_path)
    #
    # edge_index1, edge_rel1 = get_edge_info(rel_triples_1)
    # edge_index2, edge_rel2 = get_edge_info(rel_triples_2)
    #
    # tradic_edge_index1 = get_tradic_edges(edge_index1, edge_rel1, entity_cnt1, ent_map1, rel_map1)
    # tradic_edge_index2 = get_tradic_edges(edge_index2, edge_rel2, entity_cnt2, ent_map2, rel_map2)
    #
    # generate_pickle(tradic_edge_index1_path, tradic_edge_index1)
    # generate_pickle(tradic_edge_index2_path, tradic_edge_index2)

    '''
    get train test
    '''
    # ent_ill = read_idtuple_file(ent_ill_path)
    # rel_ill = read_spec_idtuple_file(rel_ill_path)
    #
    # ent_y = get_ent_aligns(ent_ill, ent_map1, ent_map2)
    # rel_y = get_ent_aligns(rel_ill, rel_map1, rel_map2)
    #
    # ent_align_num = len(ent_ill)
    # train_ey = np.array(ent_y[:ent_align_num // 10 * ent_train_split])
    # test_ey = np.array(ent_y[ent_align_num // 10 * ent_train_split:])
    #
    # rel_align_num = len(rel_ill)
    # train_ry = np.array(rel_y[:rel_align_num // 10 * rel_train_split])
    # test_ry = np.array(rel_y[rel_align_num // 10 * rel_train_split:])
    #
    # generate_pickle(ey_path, ent_y)
    # generate_pickle(ry_path, rel_y)
    #
    # generate_pickle(train_ey_path, train_ey)
    # generate_pickle(test_ey_path, test_ey)
    #
    # generate_pickle(train_ry_path, train_ry)
    # generate_pickle(test_ry_path, test_ry)

    '''
       get embedding init files
    '''

    # Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # ent2desTokens1, ent2desTokens2 = ent2desTokens_generate(Tokenizer,des_dict_path,list(index2entity1.values()),list(index2entity2.values()))
    # generate_pickle(ent2desTokens1_path, ent2desTokens1)
    # generate_pickle(ent2desTokens2_path, ent2desTokens2)
    #
    # ent2desTokens1 = process_pickle(ent2desTokens1_path)
    # ent2desTokens2 = process_pickle(ent2desTokens2_path)
    #
    # ent2tokenids1 = obj2Tokens_gene(Tokenizer, ent2desTokens1, index2entity1)
    # ent2tokenids2 = obj2Tokens_gene(Tokenizer, ent2desTokens2, index2entity2)
    # ent2data1 = obj2bert_input(index2entity1, Tokenizer, ent2tokenids1)
    # ent2data2 = obj2bert_input(index2entity2, Tokenizer, ent2tokenids2)
    #
    # generate_pickle(ent2data1_path, ent2data1)
    # generate_pickle(ent2data2_path, ent2data2)
    ent2data1 = process_pickle(ent2data1_path)
    ent2data2 = process_pickle(ent2data2_path)

    #
    # rel2tokenids1 = obj2Tokens_gene(Tokenizer, None, index2rel1, reln_max_length)
    # rel2tokenids2 = obj2Tokens_gene(Tokenizer, None, index2rel2, reln_max_length)
    #
    # reln_max_length1 = max([len(tokens) for rel_id, tokens in rel2tokenids1.items()])
    # reln_max_length2 = max([len(tokens) for rel_id, tokens in rel2tokenids2.items()])
    #
    # rel2data1 = obj2bert_input(index2rel1, Tokenizer, rel2tokenids1, False, reln_max_length1)
    # rel2data2 = obj2bert_input(index2rel2, Tokenizer, rel2tokenids2, False, reln_max_length2)
    # generate_pickle(rel2data1_path, rel2data1)
    # generate_pickle(rel2data2_path, rel2data2)
    rel2data1 = process_pickle(rel2data1_path)
    rel2data2 = process_pickle(rel2data2_path)

    '''
       get entity and relation embedding
    '''

    ENT_Model = Basic_Bert_Unit_model(768, ent_dim)
    ENT_Model.load_state_dict(torch.load(ENT_MODEL_SAVE_PATH, map_location='cpu'))
    ENT_Model.eval()
    for name, v in ENT_Model.named_parameters():
        v.requires_grad = False
    ENT_Model = ENT_Model.cuda(cuda_num)

    REL_Model = Basic_Bert_Unit_model(768, rel_dim)
    REL_Model.load_state_dict(torch.load(ENT_MODEL_SAVE_PATH, map_location='cpu'))
    REL_Model.eval()
    for name, v in REL_Model.named_parameters():
        v.requires_grad = False
    REL_Model = REL_Model.cuda(cuda_num)

    ent_emb1 = get_bert_entity_em(ent2data1, ENT_Model)
    ent_emb2 = get_bert_entity_em(ent2data2, ENT_Model)

    generate_pickle(ent_emb_path1, ent_emb1)
    generate_pickle(ent_emb_path2, ent_emb2)

    rel_emb1 = get_bert_entity_em(rel2data1, REL_Model)
    rel_emb2 = get_bert_entity_em(rel2data2, REL_Model)

    generate_pickle(rel_emb_path1, rel_emb1)
    generate_pickle(rel_emb_path2, rel_emb2)



