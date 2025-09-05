

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle

from loss import crossview_contrastive_Loss
import evaluation
from util import next_batch, next_batch_new
from fusion import MultiHeadAttention, FeedForwardNetwork
from SelfAttention import SelfAttention
from loss_new import Loss
from EWN import entropy
import scipy.io as sio

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class Encoder_share(nn.Module):
    def __init__(self,
                 share_dim,
                 activation='relu',
                 batchnorm=True):
        super(Encoder_share, self).__init__()

        self._dimshare = len(share_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        share_encoder_layers = []
        for i in range(self._dimshare):
            share_encoder_layers.append(
                nn.Linear(share_dim[i], share_dim[i + 1]))
            if i < self._dimshare - 1:
                if self._batchnorm:
                    share_encoder_layers.append(nn.BatchNorm1d(share_dim[i + 1]))
                if self._activation == 'sigmoid':
                    share_encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    share_encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    share_encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    share_encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        share_encoder_layers.append(nn.Softmax(dim=1))
        self._encodershare = nn.Sequential(*share_encoder_layers)

        # def share_encoder(self, x):
        #     latent_share = self._encodershare(x)
        #     return latent_share

    def forward(self, x):
        share_feature = self._encodershare(x)
        return share_feature

class Prediction(nn.Module):

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):

        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class Completer():

    def __init__(self,
                 config, x_train, view_num, device):

        self._config = config
        self.device = device

        # 潜在表示的特征维度
        self._latent_dim = config['Autoencoder']['arch1'][-1]
        # 双预测的潜在表示维度
        # self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        # self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch']

        # 对所有视图进行编码，或得的网络存入encoder_specific列表中
        encoder_specific=[]
        # 将所有视图融合后的编码潜在表示的特征维度
        share_dim=0
        for i in range(0, view_num):
            share_dim=x_train[i].shape[1]+share_dim
            # x_train[i].shape[1]]+config['Autoencoder']['arch'] 每个视角的编码层数 例如第一个试图的特征维度是40，则编码为[40,1024,1024,1024,128]
            self.autoencoder = Autoencoder([x_train[i].shape[1]]+config['Autoencoder']['arch'+str(i+1)],config['Autoencoder']['activations'],
                                           config['Autoencoder']['batchnorm'])
            # 把每个视图的编码模型放入列表中
            encoder_specific.append(self.autoencoder)
        self.encoder_specific=encoder_specific

        # 所有视角的统一编码
        self.share_encoder = Encoder_share([share_dim]+config['Encoder_share']['arch'],config['Encoder_share']['activations'],
                                           config['Encoder_share']['batchnorm'])
        # 预测网络的编码

        pre_specific=[]
        # 特殊预测统一
        for j in range(view_num):
            self.img2txt = Prediction([self._latent_dim] + self._config['Prediction']['arch'+str(i+1)])
            pre_specific.append(self.img2txt)
        self.pre_specific=pre_specific

        # 统一预测特殊
        self.txt2img = Prediction([self._latent_dim] + self._config['Prediction']['arch'])

        # 融合模块
        self.attention_net = MultiHeadAttention(config['training']['hidden_dim'],config['training']['attention_dropout_rate'], config['training']['num_heads'],
                                           config['training']['attn_bias_dim'])
        self.p_net = FeedForwardNetwork(view_num, config['training']['ffn_size'], config['training']['attention_dropout_rate'])

        # 损失模块
        


    def to_device(self, device):

        for model in self.encoder_specific:
            model.to(device)
        self.share_encoder.to(device)
        # self.img2txt.to(device)
        for pre_model in self.pre_specific:
            pre_model.to(device)
        self.txt2img.to(device)
        self.attention_net.to(device)
        self.p_net.to(device)
        # self.cri_loss.to(device)

    def train(self, config, logger, x_train,  Y_list, view_num, mask,  optimizer, device, p_sample, adaptive_weight, cri_loss,results,data_seed):

        flag = (torch.LongTensor([1] * view_num).to(device) == mask).int()
        # 判断相同样本的所有视角是不是都缺失
        if view_num == 2:
            flag = ( flag[:, 0] + flag[:, 1]) == view_num
        elif view_num == 3:
            flag = (flag[:, 0] + flag[:, 1] + flag[:, 2]) == view_num
        elif view_num == 4:
            flag = (flag[:, 0] + flag[:, 1] + flag[:, 2]  + flag[:, 3]) == view_num
        elif view_num == 5:
            flag = (flag[:, 0] + flag[:, 1] + flag[:, 2] + flag[:, 3]+flag[:, 4]) == view_num
        else:
            flag = ( flag[:, 0]+ flag[:, 1] + flag[:, 2] + flag[:, 3]+flag[:, 4] + flag[:, 5]) == view_num

        train_view=[]
        # 从训练数据中把缺失的数据剔除掉
        for j in range(0, view_num):
            train_view.append(x_train[j][flag])
        # cri_loss = torch.nn.MSELoss()
        # results = {'Epoch': [],'ACC': [], 'NMI': [], 'ARI': [],'purity': [], 'precision': [], 'recall': [],'f_measure': [], 'Loss': []}
        for epoch in range(config['training']['epoch']):
            # 对训练的视角进行扰动
            X = [shuffle(item) for item in train_view]
            # 损失的中间变量recon_loss  cl_loss pre_loss
            recon_loss = 0; cl_loss = 0; pre_loss = 0
            # 总体损失loss_all，重建损失loss_rec 对比损失loss_cl 预测损失loss_pre
            loss_all, loss_rec, loss_cl, loss_pre = 0, 0, 0, 0
            z_v_tsne = []
            # 分批次进行训练，每一个批次256个样本
            for batch_x,  batch_num in next_batch_new(X,  config['training']['batch_size']):
                # 将每一个批次的样本融合起来
                batch_list=[]
                for i in range(0, view_num):
                    batch_list = [batch_x[i]]+batch_list
                # 所有样本的融合 batch_share
                batch_share = torch.cat(batch_list, dim=-1)
                # 对所有的样本进行编码，获得统一潜在表示U
                U = self.share_encoder(batch_share.float())
                # 计算每个特定视角和统一表示之间的损失
                for v in range(0, view_num):
                    # 对第v个视角进行编码得到潜在表示z_v
                    z_v = self.encoder_specific[v].encoder(batch_x[v])

                    # 自注意力 
                    z_v_attention = z_v.unsqueeze(0)
                    self_dim = z_v_attention.shape[2]
                    net_v = SelfAttention(self_dim, self_dim, self_dim).to(device)
                    z_v_attention = net_v.forward(z_v_attention).to(device)
                    z_v_attention = z_v_attention.reshape(config['training']['batch_size'], self_dim)

                    z_v_tsne.append(z_v_attention)

                    # 对特定视图的潜在表示z_v进行解码得到z_v_hat
                    z_v_hat = self.encoder_specific[v].decoder(z_v_attention)

                    # 第v个视图的原始数据和z_hat之间做均方损失
                    recon_v = F.mse_loss(z_v_hat, batch_x[v])

                    recon_loss += recon_v

                    # w1, w2 = entropy(z_v, U)
                    # z_v_fea, U_fea = w1 * z_v, w2 * U
                    
                    # 每一个潜在表示z和统一潜在表示u做对比损失
                    # cl_loss_v = crossview_contrastive_Loss(z_v_fea, U_fea, config['training']['alpha'])
                    cl_loss_v = crossview_contrastive_Loss(z_v, U, config['training']['alpha'])
                    # 所有视角的对比损失
                    cl_loss += cl_loss_v
                    # cl_loss = 0
                    # 特定视图的潜在表示z_v预测统一表示U
                    img2txt_v, _ = self.pre_specific[v](z_v)
                    # img2txt, _ = self.img2txt(z_v)
                    # 统一潜在表示U预测特有的潜在表示z
                    txt2img, _ = self.txt2img(U)
                    # 二者之间做损失
                    # pre_U2z = cri_loss.forward_feature(img2txt_v, U)
                    # pre_z2U = cri_loss.forward_feature(txt2img, z_v)
                    pre_U2z = F.mse_loss(img2txt_v, U)
                    pre_z2U = F.mse_loss(txt2img, z_v)
                    # 预测损失
                    pre_loss += pre_U2z + pre_z2U
                    # pre_loss = 0
                # loss = recon_loss* config['training']['lambda2']
                # loss = cl_loss.detach()
                loss = cl_loss + recon_loss * config['training']['lambda2']

                # 开始训练双预测的的模型
                if epoch >= config['training']['start_dual_prediction']:
                        # loss += dualprediction_loss * config['training']['lambda1']
                        # loss += np.sum(pre_loss) * config['training']['lambda1']
                    loss += pre_loss * config['training']['lambda1']
                    # loss += 0

                # 所有参数清零，梯度，权重
                optimizer.zero_grad()
                loss1 = loss.detach_().requires_grad_(True)
                loss1.backward()
                    # loss.backward()
                optimizer.step() 

                    # loss_all += loss.item()
                loss_all += loss1.item()
                loss_rec += recon_loss.item()
                loss_cl += cl_loss.item()
                # loss_cl = 0 
                loss_pre += pre_loss.item()


            if (epoch + 1) % config['print_num'] == 0:
                # output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                #          "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} ===> Loss = {:.4e}" \
                #     .format((epoch + 1), config['training']['epoch'], loss_rec1, loss_rec2, loss_pre, loss_cl, loss_all)
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}" \
                         "===> Contrastive loss = {:.4e} ===> Dual prediction loss = {:.4f}  ===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec, loss_cl, loss_pre,  loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            # if (epoch + 1) % config['print_num'] == 0:
            #     scores = self.evaluation(config, logger, mask, x1_train, x2_train, Y_list, device)
            # if (epoch + 1) % config['print_num'] == 0:
            # if epoch == 20 or epoch == 50 or epoch == 100 or epoch == 200 or epoch == 300 or epoch == 400 or epoch == 499:
            #     scores = self.evaluation(config, logger, mask, x_train, Y_list, device, view_num, p_sample, adaptive_weight,epoch,loss_all,results,data_seed)
            scores = self.evaluation(config, logger, mask, x_train, Y_list, device, view_num, p_sample, adaptive_weight,epoch,loss_all,results,data_seed)
        # {'kmeans': {'AMI': 0.0163, 'NMI': 0.0427, 'ARI': 0.006, 'accuracy': 0.287, 'precision': 0.2206, 'recall': 0.2327, 'f_measure': 0.213, 'purity': 0.4696}}
        return scores['kmeans']['accuracy'], scores['kmeans']['NMI'], scores['kmeans']['ARI'],scores['kmeans']['purity'],scores['kmeans']['precision'],scores['kmeans']['recall'],scores['kmeans']['f_measure']

    def evaluation(self, config, logger, mask, x_train, Y_list, device, view_num, p_sample, adaptive_weight,epoch,loss_all,results,data_seed):
        with torch.no_grad():

            # 所有模型清零
            self.share_encoder.eval()
            for i in range(view_num):
                self.encoder_specific[i].eval()
                self.pre_specific[i].eval()
            # self.img2txt.eval(), 
            self.txt2img.eval()

            # 将所有样本融合2386，前面训练是为不缺的样本，这里包括缺失的
            all_sample = []
            for i in range(view_num):
                all_sample = [x_train[i]] + all_sample
            all_sample = torch.cat(all_sample, dim=-1)

            # 不缺失缺失数据的索引
            no_miss_idx=[]
            # 缺失数据的索引
            miss_idx=[]
            # 不缺失数据编码后的潜在表示 -每个视角
            no_miss_latent = []
            # 不缺失数据编码后的潜在表示 -所用视角
            no_miss_latent_share = []

            for v in range(view_num):
                temp1 = mask[:, v] == 1
                no_miss_idx.append(temp1)
                temp2 =mask[:, v] == 0
                miss_idx.append(temp2)

                # 对每个视图的不缺失的样本进行编码
                no_miss_latent.append(self.encoder_specific[v].encoder(x_train[v][no_miss_idx[v]]))

                # all_sample[no_miss_idx[v]] 去除每个视角缺失的样本 即每个视图不缺失的样本对应所用样本中的不缺失样本
                no_miss_latent_share.append(self.share_encoder(all_sample[no_miss_idx[v]]))

            # 初始化每个视角的潜在表示  和   每个视角与所有视角的统一表示  2386
            latent_specific=[]
            # 初始化每个视角与所有视角的统一表示  2386
            latent_share=[]
            for i in range(view_num):
                latent_specific.append(torch.zeros(x_train[i].shape[0], config['Autoencoder']['arch1'][-1]).to(device))
                latent_share.append(torch.zeros(all_sample.shape[0], config['Autoencoder']['arch1'][-1]).to(device))

            # 用一个视角缺失的样本索引 除去另一个视角中的样本，并进行编码
            latent_code_specific=[]
            latent_code_share = []
            recovery_specific = []
            recovery_share = []

            for j in range(view_num):
                latent_code_specific.append(self.share_encoder(all_sample[miss_idx[j]]))
                latent_code_share.append(self.encoder_specific[j].encoder(x_train[j][miss_idx[j]]))

                t1, _ = self.txt2img(latent_code_specific[j])
                t2, _ = self.pre_specific[j](latent_code_share[j])
                # t2, _ = self.img2txt(latent_code_share[j])

                recovery_specific.append(t1)
                recovery_share.append(t2)

            latent_fusion=[]
            for k in range(view_num):

                latent_specific[k][miss_idx[k]]=recovery_specific[k]
                latent_share[k][miss_idx[k]]=recovery_share[k]

                latent_specific[k][no_miss_idx[k]]=no_miss_latent[k]
                latent_share[k][no_miss_idx[k]]=no_miss_latent_share[k]


                latent_fusion.append(torch.cat([latent_specific[k], latent_share[k]], dim=1).cpu().numpy())

            # 对预测后的所有视图 进行融合
            hs_tensor = torch.tensor([]).to(device)
            for v in range(view_num):
                # latent_fusion = [torch.from_numpy(tensor) for tensor in latent_fusion]
                latent_fusion_gpu=torch.from_numpy(latent_fusion[v]).to(device)
                hs_tensor = torch.cat((latent_fusion_gpu, hs_tensor), 0)
            hs_tensor = torch.tensor([]).to(device)

            for v in range(view_num):
                aa = torch.from_numpy(latent_fusion[v]).to(device)
                hs_tensor = torch.cat((hs_tensor, torch.mean(aa, 1).unsqueeze(1)), 1)  # d * v

            hs_tensor = hs_tensor.t()
            # process by the attention
            hs_atten = self.attention_net(hs_tensor, hs_tensor, hs_tensor)  # v * 1

            p_learn = self.p_net(p_sample)  # v * 1

            # regulatory factor
            r = hs_atten * p_learn
            s_p = nn.Softmax(dim=0)
            r = s_p(r)

            # adjust adaptive weight
            adaptive_weight = r * adaptive_weight

            # obtain fusion feature
            fusion_feature = torch.zeros([latent_fusion[0].shape[0], latent_fusion[0].shape[1]]).to(device)


            for v in range(view_num):
                aa = torch.from_numpy(latent_fusion[v]).to(device)
                fusion_feature = fusion_feature + adaptive_weight[v].item() * aa
            fusion_feature_cpu = fusion_feature.cpu().numpy()
            scores = evaluation.clustering([fusion_feature_cpu], Y_list[0])

            # latent_fusion_tensors = [torch.from_numpy(tensor) for tensor in latent_fusion]
            # latent=torch.cat(latent_fusion_tensors, dim=1).cpu().numpy()
            # scores = evaluation.clustering([latent], Y_list[0])

            scores['kmeans']['accuracy'], scores['kmeans']['NMI'], scores['kmeans']['ARI'],scores['kmeans']['purity'],scores['kmeans']['precision'],scores['kmeans']['recall'],scores['kmeans']['f_measure']

            file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/convergence/cescnew_convergence.mat'
            results['Epoch'].append(int(epoch+1)+500*data_seed)
            results['ACC'].append(scores['kmeans']['accuracy'])
            results['NMI'].append(scores['kmeans']['NMI'])
            results['ARI'].append(scores['kmeans']['ARI'])
            results['purity'].append(scores['kmeans']['purity'])
            results['precision'].append(scores['kmeans']['precision'])
            results['recall'].append(scores['kmeans']['recall'])
            results['f_measure'].append(scores['kmeans']['f_measure'])
            results['Loss'].append(loss_all)

            # data = {'ACC':scores['kmeans']['accuracy'],'NMI':scores['kmeans']['NMI'],
            # 'ARI':scores['kmeans']['ARI'],'purity':scores['kmeans']['purity'],
            # 'precision':scores['kmeans']['precision'],'recall':scores['kmeans']['recall'],
            # 'f_measure':scores['kmeans']['f_measure'],'Loss':loss_all}

            sio.savemat(file_name,results)



            if epoch == 20:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_20.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)
            if epoch == 50:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_50.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)
            if epoch == 100:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_100.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)
            if epoch == 200:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_200.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)
            if epoch == 300:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_300.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)
            if epoch == 400:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_400.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)
            if epoch == 499:
                file_name = '/public/home/jiangli2/Complete_multiview/completer_fusion/tsne_new/STL_epoch_499.mat'
                data = {'X':fusion_feature_cpu,'Y':Y_list[0]}
                sio.savemat(file_name,data)

            logger.info("\033[2;29m" + 'view_concat ' + str(scores) + "\033[0m")

            self.share_encoder.train()
            for i in range(view_num):
                self.encoder_specific[i].train()
                self.pre_specific[i].train()
            # self.img2txt.train(), 
            self.txt2img.train()

        return scores

