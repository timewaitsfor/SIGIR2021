from utils import *
from helper import *
from param import *

from models import DGMC, RelCNN, CompGNN, Basic_Bert_Unit_model
import numpy as np

def train():
    model.train()
    optimizer.zero_grad()

    e_, eS_L, r_, rS_L = model(ent_emb1, rel_emb1, tradic_edge_index1, None, None, ent_emb2, rel_emb2,
                               tradic_edge_index2, None, None, train_ey, train_ry, test_ey, test_ry)

    loss = model.loss(eS_L, rS_L, train_ey, train_ry)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test():
    model.eval()

    e_, eS_L, r_, rS_L = model(ent_emb1, rel_emb1, tradic_edge_index1, None, None, ent_emb2, rel_emb2,
                               tradic_edge_index2, None, None, train_ey, train_ry, test_ey, test_ry)

    ehits1, rhits1, dgmc_e1_to_e2_scores = model.acc(eS_L, rS_L, test_ey, test_ry)
    ehits10, rhits10 = model.hits_at_k(10, eS_L, rS_L, test_ey, test_ry)
    # e_mrr = model.mrr(eS_L, rS_L, test_ey, test_ry)

    return ehits1, rhits1, ehits10, rhits10, dgmc_e1_to_e2_scores

if __name__ == '__main__':


    '''
    static embeddings
    '''
    ent_emb1 = process_pickle(ent_emb_path1)
    ent_emb2 = process_pickle(ent_emb_path2)

    rel_emb1 = process_pickle(rel_emb_path1)
    rel_emb2 = process_pickle(rel_emb_path2)


    '''
    fine-tune embeddings
    '''

    # print("fine-tune embeddings start...")
    # BERT_Model = Basic_Bert_Unit_model(bert_dim, ent_dim)
    # BERT_Model.cuda(cuda_num)
    #
    # ent2data1 = process_pickle(ent2data1_path)
    # ent2data2 = process_pickle(ent2data2_path)
    # ent_emb1 = get_bert_entity_em(ent2data1, BERT_Model)
    # ent_emb2 = get_bert_entity_em(ent2data2, BERT_Model)
    #
    # # rel2tokens_masks1 = process_pickle(rel2tokens_masks1_path)
    # # rel2tokens_masks2 = process_pickle(rel2tokens_masks2_path)
    # # rel_emb1 = get_bert_relation_emb(rel2tokens_masks1, BERT_Model)
    # # rel_emb2 = get_bert_relation_emb(rel2tokens_masks2, BERT_Model)
    #
    # rel2data1 = process_pickle(rel2data1_path)
    # rel2data2 = process_pickle(rel2data2_path)
    # rel_emb1 = get_bert_entity_em(rel2data1, BERT_Model)
    # rel_emb2 = get_bert_entity_em(rel2data2, BERT_Model)
    # print("fine-tune embeddings finished.")

    ent_emb1 = torch.tensor(ent_emb1, dtype=torch.float32).to(device)
    ent_emb2 = torch.tensor(ent_emb2, dtype=torch.float32).to(device)

    rel_emb1 = torch.tensor(rel_emb1, dtype=torch.float32).to(device)
    rel_emb2 = torch.tensor(rel_emb2, dtype=torch.float32).to(device)

    '''
    edge_index
    '''

    tradic_edge_index1 = process_pickle(tradic_edge_index_path1)
    tradic_edge_index2 = process_pickle(tradic_edge_index_path2)

    tradic_edge_index1 = np.array(tradic_edge_index1).T
    tradic_edge_index2 = np.array(tradic_edge_index2).T

    tradic_edge_index1 = torch.tensor(tradic_edge_index1, dtype=torch.long).to(device)
    tradic_edge_index2 = torch.tensor(tradic_edge_index2, dtype=torch.long).to(device)

    '''
    alignment ground truth
    '''

    train_ey = process_pickle(train_ey_path)
    test_ey = process_pickle(test_ey_path)

    train_ey = np.array(train_ey).T
    test_ey = np.array(test_ey).T

    train_ry = process_pickle(train_ry_path)
    test_ry = process_pickle(test_ry_path)

    train_ry = np.array(train_ry).T
    test_ry = np.array(test_ry).T

    train_ey = torch.tensor(train_ey, dtype=torch.long).to(device)
    test_ey = torch.tensor(test_ey, dtype=torch.long).to(device)

    train_ry = torch.tensor(train_ry, dtype=torch.long).to(device)
    test_ry = torch.tensor(test_ry, dtype=torch.long).to(device)

    '''
    GNN models
    '''

    psi_1 = CompGNN(ent_dim, ent_dim, num_layers, batch_norm=False, cat=True, lin=True, dropout=0.5)
    psi_2 = CompGNN(rnd_dim, rnd_dim, num_layers, batch_norm=False, cat=True, lin=True, dropout=0.0)


    '''
    main model
    '''

    model = DGMC(psi_1, psi_2, num_steps=None, k=k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('Optimize initial feature matching...')
    model.num_steps = 0
    for epoch in range(1, 201):
    # for epoch in range(1, 301):
        if epoch == 101:
        # if epoch == 151:
            print('Refine correspondence matrix...')
            model.num_steps = num_steps
            model.detach = True

        loss = train()
        if epoch % 50 == 0 or epoch > 199:
            ehits1, rhits1, ehits10, rhits10, dgmc_e1_to_e2_scores = test()
            emrr = get_mrr(dgmc_e1_to_e2_scores)
            print((f'{epoch:03d}: Loss: {loss:.4f}, eHits@1: {ehits1:.4f}, '
                   f'eHits@10: {ehits10:.4f} eMrr: {emrr:.4f} 'f'rHits@1: {rhits1:.4f} 'f'rHits@10: {rhits10:.4f}'))

