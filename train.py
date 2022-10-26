import torch

import argparse, os
import cv2
import numpy as np
from torchvision import transforms
from dataloader.dataloader import MHDataLoader
from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak
from utils.TorchLossComputer import TorchLossComputer
from dataloader.LoadVideotrain_pure import PURE_train, Normaliztion, ToTensor, RandomHorizontalFlip

from utils.TorchLossComputer import TorchLossComputer

import torch.nn as nn
import torch.optim as optim

def FeatureMap2Heatmap(x, feature1, feature2):
    ## initial images
    ## initial images
    x = x.repeat(2,1,1,1,1)
    org_img = x[0, :, 32, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_x_visual.jpg', org_img)

    ## first feature
    feature_first_frame = feature1[0, :, 16, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    heatmap = np.asarray(heatmap, dtype=np.uint8)

    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # COLORMAP_WINTER     COLORMAP_JET
    heat_img = cv2.resize(heat_img, (128, 128))
    cv2.imwrite(args.log + '/' + args.log + '_x_heatmap1.jpg', heat_img)

    ## second feature
    feature_first_frame = feature2[0, :, 8, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    heatmap = np.asarray(heatmap, dtype=np.uint8)

    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # COLORMAP_WINTER     COLORMAP_JET
    heat_img = cv2.resize(heat_img, (128, 128))
    cv2.imwrite(args.log + '/' + args.log + '_x_heatmap2.jpg', heat_img)


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))


            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# main function
def train(args):
    condition = args.version
    device_ids = args.gpu
    frames = args.frames
    batch_size = args.batch_size
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + args.log + '_con_' + str(condition) + '_log3.txt', 'a')

    for ik in range(0, 1):


        index = ik + 1

        print("cross-validastion: ", index)


        log_file.write('cross-valid : %d' % (index))
        log_file.write("\n")
        log_file.flush()

        finetune = args.finetune
        if finetune == True:
            print('finetune!\n')
            log_file.write('finetune!\n')
            log_file.flush()

            model = PhysNet_padding_ED_peak(frames = frames, device_ids = device_ids, hidden_layer = args.hidden_layer)
            
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda(device=device_ids[0])
            model.load_state_dict(torch.load('VIPL_PhysNet160_ECCV_rPPG_fold_best/VIPL_PhysNet160_ECCV_rPPG_fold_1_20.pkl'))

            lr = args.lr
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        if args.continue_train != 0:
            print('continue!\n')
            log_file.write('continue!\n')
            log_file.flush()

            model = PhysNet_padding_ED_peak(frames = frames, device_ids = device_ids, hidden_layer = args.hidden_layer)
            
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda(device=device_ids[0])
            model.load_state_dict(torch.load('./SSTTFinallog_Constrative/SSTTFinallog_Constrative_con_'+str(args.version)+'_1_'+str(args.continue_train)+'.pkl'))

            lr = args.lr
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


        else:

            print('train from scratch!\n')
            log_file.write('train from scratch!\n')
            log_file.flush()

            model = PhysNet_padding_ED_peak(frames = frames, device_ids = device_ids, hidden_layer = args.hidden_layer)
            
            model = torch.nn.DataParallel(model, device_ids=device_ids)

            model = model.cuda(device=device_ids[0])

            lr = args.lr
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


        between_loss = nn.MSELoss()
        criterion_Pearson = Neg_Pearson()

        echo_batches = args.echo_batches
        scale = args.scale

        # train
        for epoch in range(args.epochs):

            if (epoch + 1) % args.step_size == 0:
                lr *= args.gamma

            loss_rPPG_avg = AvgrageMeter()
            loss_peak_avg = AvgrageMeter()
            loss_hr_rmse = AvgrageMeter()
            loss_between_mse = AvgrageMeter()

            model.train()

            PURE_trainDL = PURE_train(scale,frames,  test=False, transform=transforms.Compose(
                [Normaliztion(), RandomHorizontalFlip(), ToTensor()]))
            dataloader_train = MHDataLoader(
                args,
                PURE_trainDL,
                batch_size=batch_size,
                shuffle=True,
                pin_memory= True
            )
            for i, sample_batched in enumerate(dataloader_train):
                inputs_1, ecg = sample_batched['video_x'].cuda(device=device_ids[0]), sample_batched['ecg'].cuda(device=device_ids[0])
                inputs_2 = sample_batched['video_y'].cuda(device=device_ids[0])
                clip_average_HR, frame_rate = sample_batched['clip_average_HR'].cuda(device=device_ids[0]), sample_batched['frame_rate'].cuda(device=device_ids[0])
                ecg_compare = torch.randn(ecg.shape[0]*2,ecg.shape[1]).cuda(device=device_ids[0])

                ecg = ecg.repeat(2,1)
                clip_average_HR = clip_average_HR.repeat(2,1)
                frame_rate = frame_rate.repeat(2,1)


                optimizer.zero_grad()

                # forward + backward + optimize
                rPPG_peak, x_visual, x_visual3232, x_visual1616 = model(inputs_1,inputs_2)
                rPPG = rPPG_peak[:, 0, :]

                rPPG_first = torch.randn(rPPG.shape[0]//2,rPPG.shape[1]).cuda(device=device_ids[0])
                rPPG_second = torch.randn(rPPG.shape[0]//2,rPPG.shape[1]).cuda(device=device_ids[0])
                for aa in range(batch_size):
                    rPPG_first[aa]=rPPG[aa]
                    rPPG_second[aa]=rPPG[aa+batch_size]
                loss_between = between_loss(rPPG_first,rPPG_second)

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2
                ecg = (ecg - torch.mean(ecg)) / torch.std(ecg) 
                loss_rPPG = criterion_Pearson(rPPG, ecg)

                clip_average_HR = (clip_average_HR - 40)  # [40, 180]
                fre_loss = 0.0
                train_rmse = 0.0

                for bb in range(inputs_1.shape[0]+inputs_2.shape[0]):
                    fre_loss_temp, train_rmse_temp = TorchLossComputer.cross_entropy_power_spectrum_loss(rPPG[bb],clip_average_HR[bb],frame_rate[bb],device_ids)
                    fre_loss = fre_loss + fre_loss_temp
                    train_rmse = train_rmse + train_rmse_temp
                fre_loss = fre_loss / (inputs_1.shape[0]+inputs_2.shape[0])
                train_rmse = train_rmse / (inputs_1.shape[0]+inputs_2.shape[0])
                
                if epoch > 20:
                    loss = loss_rPPG +fre_loss +0.1*loss_between
                else:
                    loss = loss_rPPG +fre_loss

                loss.backward()
                optimizer.step()

                n = inputs_1.size(0)+inputs_2.size(0)
                loss_rPPG_avg.update(loss_rPPG.data, n)
                loss_peak_avg.update(fre_loss.data, n)
                loss_hr_rmse.update(train_rmse, n)
                loss_between_mse.update(loss_between,n)
                if i % echo_batches == echo_batches - 1:  # print every 50 mini-batches


                    print('epoch:%d, mini-batch:%3d, lr=%f, NegPearson= %.4f, fre_CEloss= %.4f, hr_rmse= %.4f, between_mse= %.4f' % (
                    epoch + 1, i + 1, lr, loss_rPPG_avg.avg, loss_peak_avg.avg, loss_hr_rmse.avg, loss_between_mse.avg))
                    # log written
                    log_file.write(
                        'epoch:%d, mini-batch:%3d, lr=%f, NegPearson= %.4f, fre_CEloss= %.4f, hr_rmse= %.4f, between_mse= %.4f' % (
                        epoch + 1, i + 1, lr, loss_rPPG_avg.avg, loss_peak_avg.avg, loss_hr_rmse.avg, loss_between_mse.avg))
                    log_file.write("\n")
                    log_file.flush()



            log_file.write("\n")
            log_file.write('epoch:%d, mini-batch:%3d, lr=%f, NegPearson= %.4f, fre_CEloss= %.4f, hr_rmse= %.4f, between_mse= %.4f' % (
            epoch + 1, i + 1, lr, loss_rPPG_avg.avg, loss_peak_avg.avg, loss_hr_rmse.avg, loss_between_mse.avg))
            log_file.write("\n")
            log_file.write("\n")
            log_file.flush()
            scheduler.step()


            if args.continue_train != 0:
                torch.save(model.state_dict(),args.log + '/' + args.log + '_con_' + str(condition) + '_%d_%d.pkl' % (index, epoch+args.continue_train))
            else:
                torch.save(model.state_dict(),args.log + '/' + args.log + '_con_' + str(condition) + '_%d_%d.pkl' % (index, epoch))




    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=str, default=2, help='the gpu id used for predict')
    parser.add_argument('--frames',type=int, default=160,help='how many frames')
    parser.add_argument('--hidden_layer',type=int,default=128,help='how many point in hidden')    
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # default=0.0001
    parser.add_argument('--step_size', type=int, default=30,
                        help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 200
    parser.add_argument('--epochs', type=int, default=75, help='total training epochs')
    parser.add_argument('--log', type=str, default="SSTTFinallog_Constrative", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--continue_train', type = int , default=0, help='continue which epochs')
    parser.add_argument('--test', default=False, help='whether test')
    parser.add_argument('--version', default=3, help='version info')
    parser.add_argument('--n_threads', type=int, default=16,help='number of threads for data loading')
    parser.add_argument('--scale', type=str, default='', help='super resolution scale')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    args = parser.parse_args()
    if args.scale=='':
        args.scale = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
    else:
        args.scale = list(map(lambda x: float(x), args.scale.split('+')))
    if args.gpu=='':
        args.gpu = [0]
    else:
        number = args.gpu.split(',')
        args.gpu = [int(x) for x in number]

    if args.frames=='':
        args.frames = int(128)
    train(args)
