import torch
import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_auc_score


# Negative Sampling
def Sampling(pos_train_data, item_num, neg_ratio, pos_edges, neg_edges, sampling_sift_pos):
    neg_candidates = np.arange(item_num)

    if sampling_sift_pos:
        neg_items = []
        for idx, u in enumerate(pos_train_data[0]):

            length = neg_ratio - len(neg_edges[u])  

            if length > 0:
                probs = np.ones(item_num)
                probs[pos_edges[u]] = 0
                probs /= np.sum(probs)

                u_neg_items = np.array(neg_edges[u]).reshape(1,-1)
                random_negs = np.random.choice(neg_candidates, size = length, p = probs, replace = True).reshape(1, -1)
                u_neg_items = np.concatenate([u_neg_items, random_negs], axis=None).reshape(1,-1)

            else:
                u_neg_items = np.random.choice(neg_edges[u], size = neg_ratio, replace = True).reshape(1,-1)

            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis = 0) 
    
    else:
        neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace = True)
	
    neg_items = torch.from_numpy(neg_items)
    neg_items = neg_items.type(torch.int64)
	
    return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items


########################### TRAINING #####################################

def train(
    model, 
    optimizer, 
    train_loader, 
    valid_loader, 
    valid_label,
    pos_edges, 
    neg_edges, 
    params,
    device
): 
    best_auc, best_epoch, curr_acc = 0, -1, 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    # if params['enable_tensorboard']:
    #     writer = SummaryWriter()
    

    for epoch in range(params['max_epoch']):
        model.train() 
        start_time = time.time()


        for batch, x in enumerate(train_loader): # x: tensor:[users, pos_items]

            users, pos_items, neg_items = Sampling(x, params['item_num'], params['negative_num'], pos_edges, neg_edges, params['sampling_sift_pos'])
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            model.zero_grad()
            loss = model(users, pos_items, neg_items)
            train_loss = loss/params['batch_size']
            if train_loss > 10**10:
                breakpoint()

            print(f'train_loss: {train_loss}')
            # if params['enable_tensorboard']:
            #     writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        # if params['enable_tensorboard']:
        #     writer.add_scalar("Loss/train_epoch", loss, epoch)

        need_test = True
        if epoch < 50 and epoch % 5 != 0:
            need_test = False

        if need_test:
            start_time = time.time()
            acc, auc = link_test(model, valid_loader, valid_label)
            # if params['enable_tensorboard']:
            #     writer.add_scalar('Results/recall@20', Recall, epoch)
            #     writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Valid_AUC = {:.5f}, Valid_ACC : {:5f}".format(loss.item(), auc, acc))

            if auc > best_auc:
                best_auc, best_epoch, curr_acc = auc, epoch, acc
                early_stop_count = 0
                torch.save(model.state_dict(), params['model_save_path'])

            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True
        
        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best auc = {}, acc = {}'.format(best_epoch, best_auc, curr_acc))
            print('The best model is saved at {}'.format(params['model_save_path']))
            break

    # writer.flush()

    print('Training Done!')

# The below 7 functions (hit, ndcg, RecallPrecision_ATk, MRRatK_r, NDCGatK_r, test_one_batch, getLabel) follow this license.
# MIT License

# Copyright (c) 2020 Xiang Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################### TESTING #####################################
'''
Test and metrics
'''

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def RecallPrecision_ATk(test_data, r, k):
	"""
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
	right_pred = r[:, :k].sum(1)
	precis_n = k
	
	recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
	recall_n = np.where(recall_n != 0, recall_n, 1)
	recall = np.sum(right_pred / recall_n)
	precis = np.sum(right_pred) / precis_n
	return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
	"""
    Mean Reciprocal Rank
    """
	pred_data = r[:, :k]
	scores = np.log2(1. / np.arange(1, k + 1))
	pred_data = pred_data / scores
	pred_data = pred_data.sum(1)
	return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)



def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            # rating += mask[batch_users]
            
            _, rating_K = torch.topk(rating, k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        

    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG


def link_test(model, valid_loader, valid_label):
    with torch.no_grad():
        model.eval()
        # ui_emb_mul = model.test_forward()
        total_pred = []

        for batches, x in enumerate(valid_loader):
            batch_users = x[0].to(model.get_device())
            batch_items = x[1].to(model.get_device())
            pred = model.pred_link(batch_users, batch_users)
            pred = pred.detach().cpu().tolist()
            total_pred.extend(pred)

        total_pred = np.array(total_pred)
        acc = accuracy_score(valid_label, total_pred > 0.5)
        auc = roc_auc_score(valid_label, pred)

    return acc, auc

