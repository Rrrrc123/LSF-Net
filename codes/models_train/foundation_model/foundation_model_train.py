import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import importlib
from codes.utils.loss_function.common_loss_funtion import *
from codes.datas.foundation_model.r_dataset_load_mat import *
from codes.utils.log_save import *
from codes.utils.loss_function.r_simm_class import *
from codes.utils.r_common import *
os.environ["WANDB_SILENT"] = "true"


class OptimizerParamClass:
    def __init__(self):
        self.is_change_lr = 0  # 不更新学习率
        self.is_load_models = 0  # 不加载已训练的模型
        self.is_wandb = 0  # 不开启wandb
        self.lr_policy = 'lambda_exp'  # 更新学习率的方式
        self.niter = 12
        self.niter_decay = 5
        self.warmup_epochs = 10  # 学习率达到self.lr的上升轮数
        self.lr_decay = 0.99  # 学习率下降的平直程度，数值越大越平直
        self.lr = 0.001  # 最大学习率，或是恒定学习率
        self.lr_decay_iters = 5
        self.fold_num_list = ['4', '3', '2', '1', '0']  # 五折交叉验证的折数 , '1', '2', '3', '4'
        self.fold_num = "0"  # 当前折数
        self.epoch_count = 300  # 最大训练轮数
        self.batch_size = 4  # batch 数
        self.min_train_loss_l1 = 0.48
        self.task_name = "foundation_model"
        self.package_name = "codes.models." + self.task_name  # 包的路径及名称
        self.model_name = "vm_S_cnn_tf_T_mamba"  # 模块的名称实际就是.py的文件名
        self.class_name = 'MulModalityMaeMoe'  # 在.py的文件中定义的类名或是函数名
        self.modal_prob_list = [1, 0.5, 0.7, 0.3]  # 模态出现的概率[US，UE，CDFI，CEUS]

        self.datasets_path = r"D:\Dataset\key_frame_10_all_mat_5_a"  # 数据集路径
        self.models_save_path = "../../../checkpoints/" + self.task_name + "/" + self.model_name  # 模型保存路径
        self.results_save_path = "../../../results/" + self.task_name + "/" + self.model_name  # 模型记录保存路径
        self.wandb_project_name = ""  # wandb工程名
        self.models_head = ""  # 模型头字符串
        self.load_models_head = ""  # 加载的模型头字符串
        self.train_txt_file = ""  # 训练的数据集列表文件
        self.test_txt_file = ""  # 测试的数据集列表文件

    def update_opt(self):  # 更新相关五折交叉的信息。
        self.wandb_project_name = self.model_name + "_F_" + self.fold_num  # wandb工程名
        self.models_head = self.model_name + "_F_" + self.fold_num  # 模型头字符串
        self.load_models_head = self.model_name + "_F_" + self.fold_num  # 加载的模型头字符串
        self.train_txt_file = 'train_' + self.fold_num + '.txt'  # 训练的数据集列表文件
        self.test_txt_file = 'test_' + self.fold_num + '.txt'  # 测试的数据集列表文件
        if not os.path.exists(self.models_save_path):
            os.makedirs(self.models_save_path)
        if not os.path.exists(self.results_save_path):
            os.makedirs(self.results_save_path)


opt = OptimizerParamClass()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider app
        #  lying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def get_mul_modality_mae_loss(lab, out, mask, en_s_list, en_i_list):
    criterion = nn.L1Loss()
    b, c, l = en_i_list.size()
    mask_sum = sum(mask)
    # print(b, c, l)
    # sys.exit()
    loss_s = 0
    for ii in range(c):
        out_1 = out[:, :, ii * 10:(ii + 1) * 10, :, :]
        lab_1 = lab[:, :, ii * 10:(ii + 1) * 10, :, :]
        if mask[ii] == 1:
            loss_s = loss_s + criterion(out_1, lab_1)
    loss_s /= mask_sum
    en_loss = SupervisedContrastiveLoss()
    en_loss_sum = 0
    i_mask = []
    s_mask = []
    jj = 1
    en_list_i = torch.zeros(b, mask_sum, l)
    en_list_s = torch.zeros(b, mask_sum, l)
    for ii in range(c):
        if mask[ii] == 1:
            i_mask.append(ii + 1)
            s_mask.append(0)
            en_list_i[:, jj:jj + 1, :] = en_i_list[:, ii:ii + 1, :]
            en_list_s[:, jj:jj + 1, :] = en_s_list[:, ii:ii + 1, :]
            jj += 1
    mask_class = s_mask + i_mask
    en_list = torch.cat((en_list_s, en_list_i), dim=1)
    # print(mask, mask_class, en_list_i.size(), en_list_s.size())
    # sys.exit()
    for bb in range(b):
        en_loss_sum = en_loss_sum + en_loss(en_list[bb], torch.tensor(mask_class))
    # en_loss_sum /= mask_sum
    en_loss_sum /= b
    loss_sum = loss_s + en_loss_sum
    # print(mask, sum(mask), "loss:%.4f=%.4f+%.4f+%.4f+%.4f" % (
    #     loss_sum.item(), loss_s.item(), en_i_loss.item(), en_s_loss.item(), en_i_s_loss.item()))
    return loss_sum, [loss_sum.item(), loss_s.item(), en_loss_sum.item()]


def get_mul_modality_mae_moe_loss_0(lab, out, mask, en_s_list, en_i_list):
    criterion = nn.L1Loss()
    simm_loss = RSSIMClass()
    simm_loss = simm_loss.cuda()
    b, c, l = en_i_list.size()
    # print(b, c, l)
    en_s_list = en_s_list.reshape(b, c // 10, l * 10)
    en_i_list = en_i_list.reshape(b, c // 10, l * 10)
    b, c, l = en_i_list.size()
    # print(b, c, l)
    mask_sum = sum(mask)
    loss_l1 = 0
    loss_ssim = 0
    for ii in range(c):
        out_1 = out[:, :, ii * 10:(ii + 1) * 10, :, :]
        lab_1 = lab[:, :, ii * 10:(ii + 1) * 10, :, :]
        if mask[ii] == 1:
            loss_l1 = loss_l1 + criterion(out_1, lab_1)
            for jj in range(10):
                out_1_i = out_1[:, :, jj, :, :]
                lab_1_i = lab_1[:, :, jj, :, :]
                # print(out_1_i.size(), lab_1_i.size())
                loss_ssim = loss_ssim + (1 - simm_loss((out_1_i + 1) * 128, (lab_1_i + 1) * 128))
            loss_ssim /= 10
    loss_l1 /= mask_sum
    loss_ssim /= mask_sum
    # loss_l1 = loss_l1 + loss_ssim
    en_i_mat = torch.zeros((b, c, c))
    en_s_mat = torch.zeros((b, c, c))
    en_i_s_mat = torch.zeros((b, c, c))
    for ii in range(b):
        for jj in range(c):
            for kk in range(c):
                if jj != kk and mask[jj] == 1 and mask[kk] == 1:
                    if torch.sum(en_i_list[ii][jj] * en_i_list[ii][kk]) != 0:
                        en_i_mat[ii][jj][kk] = contrastive_loss(en_i_list[ii][jj], en_i_list[ii][kk], 0)
                    else:
                        en_i_mat[ii][jj][kk] = 0.0
                    if torch.sum(en_s_list[ii][jj] * en_s_list[ii][kk]) != 0:
                        en_s_mat[ii][jj][kk] = contrastive_loss(en_s_list[ii][jj], en_s_list[ii][kk], 1, margin=1.0)
                    else:
                        en_s_mat[ii][jj][kk] = 0.0
                    if torch.sum(en_i_list[ii][jj] * en_s_list[ii][kk]) != 0:
                        en_i_s_mat[ii][jj][kk] = contrastive_loss(en_i_list[ii][jj], en_s_list[ii][kk], 0)
                    else:
                        en_i_s_mat[ii][jj][kk] = 0.0

    #  print(en_i_mat[ii])
    #  print(en_s_mat[ii])
    # print(en_i_s_mat[ii])

    en_i_loss = 1 * torch.sum(en_i_mat) / (torch.count_nonzero(en_i_mat) + 1e-8)
    en_s_loss = 1 * torch.sum(en_s_mat) / (torch.count_nonzero(en_s_mat) + 1e-8)
    en_i_s_loss = 1 * torch.sum(en_i_s_mat) / (torch.count_nonzero(en_i_s_mat) + 1e-8)
    loss_sum = loss_l1 + loss_ssim + en_i_loss + en_s_loss + en_i_s_loss
    # print(mask, sum(mask), "loss:%.4f=%.4f+%.4f+%.4f+%.4f" % (
    #     loss_sum.item(), loss_s.item(), en_i_loss.item(), en_s_loss.item(), en_i_s_loss.item()))
    return loss_sum, [loss_sum.item(), loss_l1.item(), loss_ssim.item(), en_i_loss.item(), en_s_loss.item(),
                      en_i_s_loss.item()]


def get_mul_modality_mae_loss_l1(lab, out, mask):
    criterion = nn.L1Loss()
    b, c, l, _, _ = out.size()
    # print(b,c,l)
    mask_sum = sum(mask)
    loss_l1 = 0
    for ii in range(l // 10):
        out_1 = out[:, :, ii * 10:(ii + 1) * 10, :, :]
        lab_1 = lab[:, :, ii * 10:(ii + 1) * 10, :, :]
        if mask[ii] == 1:
            loss_l1 = loss_l1 + criterion(out_1, lab_1)

    loss_l1 /= mask_sum
    loss_sum = loss_l1
    # print(mask, sum(mask), "loss:%.4f=%.4f+%.4f+%.4f+%.4f" % (
    #     loss_sum.item(), loss_s.item(), en_i_loss.item(), en_s_loss.item(), en_i_s_loss.item()))
    return loss_sum, [loss_sum.item(), loss_l1.item()]


def get_scheduler():
    global epoch, optimizer, opt

    """
    scheduler definition
    :param optimizer:  原始优化器
    :param opt: 对应参数
    :return: 对应scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        # Return the LambdaLR scheduler
    elif opt.lr_policy == 'lambda_exp':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                # 预热阶段：学习率从 0 增加到 1
                lr_l = min(1, (epoch + 1) / opt.warmup_epochs)
            else:
                # 衰减阶段：指数衰减，最小学习率为 min_lr
                lr_l = max(0.02, 1.0 * (opt.lr_decay ** (epoch - opt.warmup_epochs)))
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'lambda_cosine':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                lr_l = epoch / opt.warmup_epochs
            else:
                lr_l = max(1e-5, 0.5 * \
                           (1. + math.cos(
                               math.pi * (epoch - opt.warmup_epochs) / (opt.epoch_count - opt.warmup_epochs))))
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_count, eta_min=0)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


if __name__ == '__main__':
    set_seed(1234)
    datasets = RDatasetFiveFoldOneModelityClass()
    test_datasets = RDatasetFiveFoldOneModelityClass()
    datasets.set_modality_name("ALL")
    test_datasets.set_modality_name("ALL")
    datasets_path = opt.datasets_path
    log_csv = RLogSave()
    print("训练的基础模型是：", opt.model_name)
    module = importlib.import_module(f"{opt.package_name}.{opt.model_name}")
    obj = getattr(module, opt.class_name)
    for fold_num in opt.fold_num_list:
        opt.fold_num = fold_num
        opt.update_opt()
        if opt.is_wandb == 1:
            wandb.login(key="2f4d2fd52729aa8dac770625c2657fe70f1073d5")
            wandb.init(project=opt.wandb_project,
                       config={"epochs": opt.epoch_count, "lr": opt.lr, "batch_size": opt.batch_size})
        models_save_path = opt.models_save_path
        models_head = opt.models_head
        load_models_head = opt.load_models_head
        datasets.create_dataset(datasets_path, opt.train_txt_file)
        test_datasets.create_dataset(datasets_path, opt.test_txt_file)
        print(datasets.get_dataset_count())
        print(test_datasets.get_dataset_count())

        batch_size = opt.batch_size
        train_data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=1)
        test_data_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1)

        model = obj()

        if opt.is_load_models == 1:
            model_dir = os.path.join(models_save_path, load_models_head + "_all.pth")
            print(model_dir)
            model.load_state_dict(torch.load(model_dir))
        model = model.cuda()
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        lr = opt.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9999), eps=1e-08,
                                      weight_decay=0.0001, amsgrad=False)

        num_epochs = opt.epoch_count
        epoch = 0
        min_loss = 100
        scheduler = get_scheduler()
        log_csv.clear_all()
        log_csv.set_table_head(["epoch", "loss_sum", "loss_l1", "loss_simm", "loss_i", "loss_s", "loss_i_s"])
        for epoch in range(num_epochs):
            run_loss = 0
            loss_l1 = 0
            loss_simm = 0
            loss_i = 0
            loss_s = 0
            loss_i_s = 0
            model.train()

            for i, data in enumerate(train_data_loader, 0):
                inputs = data['inputs']  # us_inputs,ue_inputs,cd_inputs,ce_inputs
                labels = data['inputs']
                inputs = torch.transpose(inputs, 1, 2)
                labels = torch.transpose(labels, 1, 2)
                inputs = inputs.cuda()
                labels = labels.cuda()
                mask_code = mask_code_generator(4, opt.modal_prob_list)
                optimizer.zero_grad()
                outputs, encode_s_list, encode_i_list, _ = model(inputs, mask_code)
                loss, loss_item_list = get_mul_modality_mae_moe_loss_0(labels, outputs, mask_code,
                                                                       encode_s_list, encode_i_list)
                if opt.is_wandb == 1:
                    wandb.log({"loss_sum": loss.item(),
                               "loss_L1": loss_item_list[1],
                               "loss_ssim": loss_item_list[2],
                               "loss_i": loss_item_list[3],
                               "loss_s": loss_item_list[4],
                               "loss_i_s": loss_item_list[5]
                               })
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
                loss_l1 += loss_item_list[1]
                loss_simm += loss_item_list[2]
                loss_i += loss_item_list[3]
                loss_s += loss_item_list[4]
                loss_i_s += loss_item_list[5]

                print("\r     >", i, run_loss, end="")
            if opt.is_change_lr == 1:
                scheduler.step()
                cur_lr = scheduler.get_last_lr()[0]
            else:
                cur_lr = lr
            print("\nlr:", cur_lr)
            train_loss = run_loss / len(train_data_loader)
            train_loss_l1 = loss_l1 / len(train_data_loader)
            train_loss_simm = loss_simm / len(train_data_loader)
            train_loss_i = loss_i / len(train_data_loader)
            train_loss_s = loss_s / len(train_data_loader)
            train_loss_i_s = loss_i_s / len(train_data_loader)

            log_row = [epoch,
                       train_loss,
                       train_loss_l1,
                       train_loss_simm,
                       train_loss_i,
                       train_loss_s,
                       train_loss_i_s]
            log_csv.add_tail(log_row)
            log_csv.save_log(opt.results_save_path, opt.models_head)

            print("epoch:", epoch,
                  "  run_loss:%.5f" % train_loss,
                  "  loss_l1:%.5f" % train_loss_l1,
                  "  loss_simm:%.5f" % train_loss_simm,
                  "  loss_i:%.5f" % train_loss_i,
                  "  loss_s:%.5f" % train_loss_s,
                  "  loss_i_s:%.5f" % train_loss_i_s,
                  )

            if opt.is_wandb == 1:
                wandb.log({"epoch:": epoch,
                           "loss_sum": train_loss,
                           "loss_L1_sum": train_loss_l1,
                           "loss_simm_sum": train_loss_simm,
                           "loss_i_sum": train_loss_i,
                           "loss_s_sum": train_loss_s,
                           "loss_i_s_sum": train_loss_i_s
                           })
            # print(train_loss, min_loss)
            if train_loss < min_loss and train_loss_l1 < opt.min_train_loss_l1:
                min_loss = train_loss
                model.eval()
                model.save_models(models_save_path, models_head + "_new_all")

            elif epoch % 10 == 0:
                model.eval()
                model.save_models(models_save_path, models_head + "_10_all")

    if opt.is_wandb == 1:
        wandb.finish()
