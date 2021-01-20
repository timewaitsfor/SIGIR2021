import torch

print("In params:")

cuda_num = 0
device = 'cuda:'+str(cuda_num) if torch.cuda.is_available() else 'cpu'

ent_dim = 300
rel_dim = 300
rnd_dim = 32
bert_dim = 768

num_layers = 3
ent_train_split = 3
rel_train_split = 3
k = 25
num_steps = 30

bert_batch_size = 64


des_limit = 128 - 2 # max length of description/name.
des_max_length = 128
reln_max_length = 64

LANG1 = 'zh' #language 'zh'/'ja'/'fr'
LANG2 = 'en'
data_dir = r"./data/{}_en/".format(LANG1)
ipt_dir = r"./data/{}_en/input/".format(LANG1)
opt_dir = r"./data/{}_en/output/".format(LANG1)

index2entity1_path = ipt_dir + "index2entity1.pkl"
index2entity2_path = ipt_dir + "index2entity2.pkl"

index2rel1_path = ipt_dir + "index2rel1.pkl"
index2rel2_path = ipt_dir + "index2rel2.pkl"

ent_emb_path1 = ipt_dir + "ent_emb1.pkl"
ent_emb_path2 = ipt_dir + "ent_emb2.pkl"

rel_emb_path1 = ipt_dir + "rel_emb1.pkl"
rel_emb_path2 = ipt_dir + "rel_emb2.pkl"


ent_em_dict_path1 = ipt_dir + "ent_em_dict1.pkl"
ent_em_dict_path2 = ipt_dir + "ent_em_dict2.pkl"

rel_em_dict_path1 = ipt_dir + "rel_emb_dict1.pkl"
rel_em_dict_path2 = ipt_dir + "rel_emb_dict2.pkl"

tradic_edge_index_path1 = ipt_dir + "tradic_edge_index1.pkl"
tradic_edge_index_path2 = ipt_dir + "tradic_edge_index2.pkl"


ent_ill_path = data_dir + 'ref_ent_ids'
rel_ill_path = data_dir + 'ref_r_ids'

train_ey_path = ipt_dir + "train_ey.pkl"
test_ey_path = ipt_dir + "test_ey.pkl"

ey_path = ipt_dir + "ey.pkl"
ry_path = ipt_dir + "ry.pkl"

train_ry_path = ipt_dir + "train_ry.pkl"
test_ry_path = ipt_dir + "test_ry.pkl"


ent_ids1_path = data_dir + "ent_ids_1"
ent_ids2_path = data_dir + "ent_ids_2"

rel_ids1_path = data_dir + "rel_ids_1"
rel_ids2_path = data_dir + "rel_ids_2"

ent2data1_path = ipt_dir+"ent2data1.pkl"
ent2data2_path = ipt_dir+"ent2data2.pkl"

rel2data1_path = ipt_dir+"rel2data1.pkl"
rel2data2_path = ipt_dir+"rel2data2.pkl"

ent2desTokens1_path = ipt_dir+"ent2desTokens1.pkl"
ent2desTokens2_path = ipt_dir+"ent2desTokens2.pkl"

rel2tokens_masks1_path = ipt_dir+"rel2tokens_masks1.pkl"
rel2tokens_masks2_path = ipt_dir+"rel2tokens_masks2.pkl"

des_dict_path = "../data/dbp15k-bert-int/2016-10-des_dict" #description data path

triples1_path = data_dir+"triples_1"
triples2_path = data_dir+"triples_2"


tradic_edge_index1_path = ipt_dir+"tradic_edge_index1.pkl"
tradic_edge_index2_path = ipt_dir+"tradic_edge_index2.pkl"



'''
bert model
'''
MARGIN = 3 # margin
LEARNING_RATE = 1e-5 # learning rate
TRAIN_BATCH_SIZE = 24
NEG_NUM = 2 # negative sample num
EPOCH_NUM = 4 #training epoch num
NEAREST_SAMPLE_NUM = 128
CANDIDATE_GENERATOR_BATCH_SIZE = 128
CUDA_NUM = 0
ENT_MODEL_SAVE_PATH = "./Save_model/" + "DBP15K_{}en_ent_BERT_MODEL.p".format(LANG1)
REL_MODEL_SAVE_PATH = "./Save_model/" + "DBP15K_{}en_rel_BERT_MODEL.p".format(LANG1)
TEST_BATCH_SIZE = 128
TOPK = 50