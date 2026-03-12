

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')
parser.add_argument('--pre_iterations', type=int, default=30000, help='预训练iter number')
parser.add_argument('--max_iterations', type=int, default=90000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--sam_batch_size', type=int, default=8)
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--image_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=3, help='output channel of network') # include background
parser.add_argument('--crop_method', type=str, default='center', help='center random or random crop')
parser.add_argument('--min_pixel_num_obj', type=int, default=16, help='i')
parser.add_argument('--min_total_pixel_num', type=int, default=36)

# sam-fintune
parser.add_argument("--sam_model_type", type=str, default="vit_b", help="sam model type")
parser.add_argument('--sam_lr', type=float, default=1e-4, help='segmentation network learning rate')
parser.add_argument("--sam_fm", type=bool, default=True, help="mix forward class image")

# label and unlabel
parser.add_argument('--use_union_loss', type=bool, default=True, help='loss=loss1+loss2')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=14, help='labeled data')  # 有标签数据占比7-10%,14-20%
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')

# num worker
parser.add_argument('--train_num_workers', type=int, default=12, help='train seg num_workers')

# pretrain
parser.add_argument('--only_weak', type=bool, default=False, help='only use weak data')

# selftrain
parser.add_argument('--use_mt', type=str, default='unet', help='use mean teacher create Pseudo label')
parser.add_argument('--use_cross_loss', type=bool, default=True, help='use cross model Pseudo label loss')
parser.add_argument("--sam_fussion_type", type=str, default="time_sum", help="concat, avg_sum, time_sum")

# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

# self retrain
parser.add_argument('--use_early_stop', type=bool, default=True, help='self training early stop')

args = parser.parse_args()
resizeLongestSide = ResizeLongestSide(1024)


def cal_sam_img_encoder(model, new_img):
    model.eval()
    with torch.no_grad():
        img_target_size = resizeLongestSide.get_preprocess_shape(*np.array(new_img[0].squeeze().shape), 1024)
        new_img = new_img.repeat(1, 3, 1, 1)
        new_img = F.interpolate(new_img, size=img_target_size, mode='bilinear', align_corners=False)
        out_ef_sam = model.image_encoder(new_img.cuda())
        del new_img
        torch.cuda.empty_cache()
    return out_ef_sam

def cal_time_sum_weight(current_iter_num, max_iterations):

    T = int(max_iterations * 4 // 5)

    if current_iter_num < T:
        ratio = current_iter_num / T
        sam_weight = 1.0 - ratio
        unet_weight = ratio
    else:
        sam_weight = 0.0
        unet_weight = 1.0

    return [sam_weight, unet_weight]

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_pseudo_labels(logits_1, logits_2, min_pixel_num_obj, num_classes):

    assert logits_1.shape == logits_2.shape

    prob1 = torch.softmax(logits_1, dim=1)
    prob2 = torch.softmax(logits_2, dim=1)
    pseudo_labels = torch.argmax(0.5 * (prob1 + prob2), dim=1).detach().cpu()

    pseudo_labels = get_2DLargestCC(pseudo_labels, num_classes, min_pixel_num_obj)

    return pseudo_labels



def generate_ulab_by_model(uimg_w, model1, model2, min_pixel_num_obj=0, num_classes=3, out_ef_sams=None, sam_sum_weight=[-1, -1]):
    model1.eval()
    model2.eval()

    with torch.no_grad():
        uimg_w = uimg_w.unsqueeze(1).cuda()
        out1_logit_uw = model1(uimg_w, out_ef_sams=out_ef_sams, sam_sum_weight=sam_sum_weight)
        out2_logit_uw = model2(uimg_w, out_ef_sams=out_ef_sams, sam_sum_weight=sam_sum_weight)
        pseudo_ulab_w = generate_pseudo_labels(out1_logit_uw, out2_logit_uw, min_pixel_num_obj, num_classes)

        # if model_sam is None:
        pseudo_ulab = pseudo_ulab_w.detach().cpu()

        del pseudo_ulab_w, out2_logit_uw, out1_logit_uw
        torch.cuda.empty_cache()

        return pseudo_ulab


def generate_ulab_by_teacher_model(uimg_w, model, min_pixel_num_obj, num_classes, out_ef_sams=None, sam_sum_weight=[-1, -1]):
    model.eval()

    with torch.no_grad():
        uimg_w = uimg_w.unsqueeze(1).cuda()

        out_logit_uw = model(uimg_w, out_ef_sams=out_ef_sams, sam_sum_weight=sam_sum_weight)
        pseudo_ulab_w = torch.argmax(torch.softmax(out_logit_uw, dim=1), dim=1).detach().cpu()
        pseudo_ulab_w = get_2DLargestCC(pseudo_ulab_w, num_classes, min_pixel_num_obj)

        # if model_sam is None:
        pseudo_ulab = pseudo_ulab_w.detach().cpu()

        return pseudo_ulab


def pre_train_unet(args, ad_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    total_iterations = args.pre_iterations + args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model1 = AD_net(in_chns=1, class_num=num_classes, model="UNet_fm")
    model2 = AD_net(in_chns=1, class_num=num_classes, model="ResUNet_fm")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, labeled_num=args.labeled_num, transform=transforms.Compose([WeakStrongAugment_fm(args.image_size)]))  # 256*256
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=lambda batch: pre_random_pad_collate(batch, db_train, args.batch_size))
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model1.train()
    model2.train()

    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=args.num_classes)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].squeeze(), sampled_batch['label']
            indices = np.random.permutation(len(volume_batch))
            volume_batch = volume_batch[indices]
            label_batch = label_batch[indices]

            img_a_w, img_b_w = volume_batch[:args.labeled_bs], volume_batch[args.labeled_bs:]
            lab_a_w_s, lab_b_w_s = label_batch[:args.labeled_bs], label_batch[args.labeled_bs:]

            volume_batch_strong = sampled_batch['image_strong'].squeeze()
            volume_batch_strong = volume_batch_strong[indices]
            img_a_s, img_b_s = volume_batch_strong[:args.labeled_bs], volume_batch_strong[args.labeled_bs:]

            forward_mask_a_w_s = lab_a_w_s > 0
            forward_mask_b_w_s = lab_b_w_s > 0

            new_imgs = []
            new_labs = []

            for i in range(len(img_a_w)):
                if not args.only_weak:
                    new_img, new_lab = create_new_img_lab(imgs=torch.cat((img_a_w[i].unsqueeze(0), img_a_s[i].unsqueeze(0), img_b_w[i].unsqueeze(0), img_b_s[i].unsqueeze(0)), dim=0), labs=torch.cat((lab_a_w_s[i].unsqueeze(0), lab_a_w_s[i].unsqueeze(0), lab_b_w_s[i].unsqueeze(0), lab_b_w_s[i].unsqueeze(0)), dim=0), fms=torch.cat((forward_mask_a_w_s[i].unsqueeze(0), forward_mask_a_w_s[i].unsqueeze(0), forward_mask_b_w_s[i].unsqueeze(0), forward_mask_b_w_s[i].unsqueeze(0)), dim=0), min_total_pixel_num=args.min_total_pixel_num, crop_method=args.crop_method)
                else:
                    new_img, new_lab = create_new_img_lab(imgs=torch.cat((img_a_w[i].unsqueeze(0), img_b_w[i].unsqueeze(0)), dim=0), labs=torch.cat((lab_a_w_s[i].unsqueeze(0), lab_b_w_s[i].unsqueeze(0)), dim=0), fms=torch.cat((forward_mask_a_w_s[i].unsqueeze(0), forward_mask_b_w_s[i].unsqueeze(0)), dim=0), min_total_pixel_num=args.min_total_pixel_num, crop_method=args.crop_method)
                new_imgs.append(new_img)
                new_labs.append(new_lab)

            new_imgs = torch.concat(new_imgs, dim=0)
            new_labs = torch.concat(new_labs, dim=0)

            new_imgs, new_labs = new_imgs.cuda(), new_labs.cuda()


            predict_probs1 = model1(new_imgs)  # unet
            predict_probs2 = model2(new_imgs)  # resunet

            loss1 = ce_loss(predict_probs1, new_labs.long()) + dice_loss(torch.softmax(predict_probs1, dim=1), new_labs.unsqueeze(1)) # avg_loss-false
            loss2 = ce_loss(predict_probs2, new_labs.long()) + dice_loss(torch.softmax(predict_probs2, dim=1), new_labs.unsqueeze(1))

            if args.use_union_loss:
                loss = loss1 + loss2
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
            else:
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

            iter_num = iter_num + 1
            lr_ = args.base_lr * (1.0 - iter_num / total_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            if args.use_union_loss:
                logging.info('iteration: %d, loss_total: %f, loss_unet: %f, loss_resunet: %f' % (iter_num, loss, loss1, loss2))
            else:
                logging.info('iteration: %d, loss_unet: %f, loss_resunet: %f' % (iter_num, loss1, loss2))

            if (iter_num > 0 and iter_num % 3000 == 0) or (iter_num == max_iterations - 1):
                model1.eval()
                model2.eval()
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i_1 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.image_size)
                    metric_i_2 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.image_size)
                    metric_list_1 += np.array(metric_i_1)
                    metric_list_2 += np.array(metric_i_2)

                metric_list_1 = metric_list_1 / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)

                performance1 = np.mean(metric_list_1, axis=0)
                performance2 = np.mean(metric_list_2, axis=0)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_best_path = os.path.join(ad_path, 'best_model_unet.pth')
                    save_net_opt(model1, optimizer1, save_best_path)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_best_path = os.path.join(ad_path, 'best_model_resunet.pth')
                    save_net_opt(model2, optimizer2, save_best_path)

                logging.info('unet iteration: %d, mean_dice : %f, best_val_maxdice : %f' % (iter_num, performance1, best_performance1))
                logging.info('resnet iteration: %d, mean_dice : %f, best_val_maxdice : %f' % (iter_num, performance2, best_performance2))
                model1.train()
                model2.train()
        if iter_num >= max_iterations:
            iterator.close()
            break

def pre_finetune_sam(args, pre_fitune_sam_path):
    base_lr = args.sam_lr
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs = args.sam_batch_size // 2

    model_sam = sam_model_registry[args.sam_model_type](args.sam_checkpoint).cuda()
    for param in model_sam.parameters():
        param.requires_grad = True

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num, sam_fintune=True)
    print("labeled slices is:{}".format(labeled_slice))

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, labeled_num=args.labeled_num, transform=transforms.Compose([NoAugment_fm(args.image_size)]))  # 256*256
    trainloader = DataLoader(db_train, batch_size=args.sam_batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=lambda batch: pre_random_pad_collate(batch, db_train, args.sam_batch_size))
    optimizer = optim.SGD(model_sam.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model_sam.train()

    logging.info("Start SAM fituning")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # loss functions
    criterion_loss = losses.FocalDiceloss() 

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].squeeze(), sampled_batch['label']
            if args.sam_fm:
                img_a = volume_batch[:labeled_sub_bs]
                img_b = volume_batch[labeled_sub_bs:]
                lab_a = label_batch[:labeled_sub_bs]
                lab_b = label_batch[labeled_sub_bs:]
                forward_mask_a = lab_a > 0
                forward_mask_b = lab_b > 0
                volume_temps = []
                label_temps = []
                for i in range(len(img_a)):
                    volume_temp, label_temp = create_new_img_lab(imgs=torch.cat((img_a[i].unsqueeze(0), img_b[i].unsqueeze(0)), dim=0), labs=torch.cat((lab_a[i].unsqueeze(0), lab_b[i].unsqueeze(0)), dim=0), fms=torch.cat((forward_mask_a[i].unsqueeze(0), forward_mask_b[i].unsqueeze(0)), dim=0), min_total_pixel_num=args.min_total_pixel_num, crop_method=args.crop_method)
                    volume_temps.append(volume_temp.squeeze(1))
                    label_temps.append(label_temp)
                volume_batch = torch.cat(volume_temps, dim=0)
                label_batch = torch.cat(label_temps, dim=0)

            predict_masks = []
            iou_predictions = []
            obj_labels = []

            for i in range(len(volume_batch)):
                input_image = volume_batch[i].float().unsqueeze(0).unsqueeze(0)
                img_target_size = resizeLongestSide.get_preprocess_shape(*np.array(volume_batch[i].shape), long_side_length=1024)
                input_image = input_image.repeat(1, 3, 1, 1)
                input_image = F.interpolate(input_image, size=img_target_size, mode='bilinear', align_corners=False).cuda()
                image_embedding = model_sam.image_encoder(input_image)
                del input_image
                centeries, bboxes, label = cal_center_bbox_category_from_label(label_batch[i])
                if centeries is None or centeries.numel() == 0:
                    continue
                obj_labels.append(label)
                del label

                obj_num = centeries.size()[0]
                prompt_points = centeries.unsqueeze(0).repeat(obj_num, 1, 1).cuda()
                transformed_point_coords = resizeLongestSide.apply_coords_torch(prompt_points, args.image_size)
                del prompt_points
                prompt_points_label = []

                for j in range(obj_num):
                    prompt_label = [0] * obj_num
                    prompt_label[j] = 1
                    prompt_points_label.append(prompt_label)

                point_labels = torch.as_tensor(np.array(prompt_points_label), dtype=torch.int).cuda()
                points = (transformed_point_coords, point_labels)
                del point_labels

                input_boxes = bboxes.cuda()
                boxes = resizeLongestSide.apply_boxes_torch(input_boxes, args.image_size)
                del input_boxes

                sparse_embeddings, dense_embeddings = model_sam.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=None,
                )
                del points, boxes

                masks, iou_prediction = model_sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=model_sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                del image_embedding, sparse_embeddings, dense_embeddings
                if masks.shape[-2] != args.image_size[0] or masks.shape[-1] != args.image_size[1]:
                    masks = F.interpolate(masks, (args.image_size[0], args.image_size[1]), mode="bilinear", align_corners=False)
                predict_masks.append(masks)
                iou_predictions.append(iou_prediction)
                del masks, iou_prediction

            if len(predict_masks) == 0:
                continue

            predict_masks = torch.cat(predict_masks, dim=0)
            iou_predictions = torch.cat(iou_predictions, dim=0)
            obj_labels = torch.cat(obj_labels, dim=0).unsqueeze(dim=1).cuda()

            loss = criterion_loss(predict_masks, obj_labels, iou_predictions)
            del predict_masks, obj_labels, iou_predictions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            lr_ = args.sam_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            logging.info('iteration: %d, loss_sam: %f' % (iter_num, loss))

            if (iter_num == max_iterations) or (iter_num == max_iterations - 1):
                save_final_model_sam_path = os.path.join(pre_fitune_sam_path, f"fintuned_sam_{args.sam_model_type.split('_')[-1]}_fm.pth")
                save_net_opt(model_sam, optimizer, save_final_model_sam_path)

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

def self_train(args, ad_path):
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    start_iterations = args.pre_iterations
    total_iterations = args.pre_iterations + args.max_iterations
    base_lr = args.base_lr * (1.0 - start_iterations / total_iterations) ** 0.9
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    pre_trained_model1 = 'model_unet.pth'
    pre_trained_model2 = 'model_resunet.pth'

    if len(args.sam_checkpoint) != 0:
        model_sam = sam_model_registry[args.sam_model_type](args.sam_checkpoint).cuda()
        for param in model_sam.parameters():
            param.requires_grad = False
        model_sam.eval()
        if 'fintuned' in args.sam_checkpoint:
            if args.sam_fm:
                ad_path = os.path.join(ad_path, 'fintuned_fm_sam')
            else:
                ad_path = os.path.join(ad_path, 'fintuned_sam')
        else:
            ad_path = os.path.join(ad_path, 'ori_sam')
        if args.sam_fussion_type == 'concat':
            sam_sum_weight = [-1, -1]
        elif args.sam_fussion_type == 'avg_sum':
            sam_sum_weight = [0.5, 0.5]
    else:
        model_sam = None
        ad_path = os.path.join(ad, 'no_sam')

    if model_sam is None:
        model1 = AD_net(in_chns=1, class_num=num_classes, model="UNet_fm")  # unet
        model2 = AD_net(in_chns=1, class_num=num_classes, model="ResUNet_fm")  # resunet
    else:
        model1 = AD_net(in_chns=1, class_num=num_classes, model="UNet_fm", sam_fussion_type=args.sam_fussion_type)  # unet
        model2 = AD_net(in_chns=1, class_num=num_classes, model="ResUNet_fm", sam_fussion_type=args.sam_fussion_type)  # resunet

    if args.use_mt == 'unet':
        if model_sam is None:
            ema_model = AD_net(in_chns=1, class_num=num_classes, model="UNet_fm", ema=True)
        else:
            ema_model = AD_net(in_chns=1, class_num=num_classes, model="UNet_fm", sam_fussion_type=args.sam_fussion_type, ema=True)
    elif args.use_mt == 'res_unet':
        if model_sam is None:
            ema_model = AD_net(in_chns=1, class_num=num_classes, model="ResUNet_fm", ema=True)
        else:
            ema_model = AD_net(in_chns=1, class_num=num_classes, model="ResUNet_fm", sam_fussion_type=args.sam_fussion_type, ema=True)

    if not os.path.exists(ad_path):
        os.makedirs(ad_path)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, labeled_num=args.labeled_num, transform=transforms.Compose([WeakStrongAugment_fm(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Train labeled {} samples".format(labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=args.train_num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(model1, pre_trained_model1)  # unet
    load_net(model2, pre_trained_model2)  # resunet
    if args.use_mt == 'unet':
        load_net(ema_model, pre_trained_model1)
        logging.info("Loaded from {}".format(pre_trained_model1))
    elif args.use_mt == 'res_unet':
        load_net(ema_model, pre_trained_model2)
        logging.info("Loaded from {}".format(pre_trained_model2))
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # loss functions
    ce_loss = CrossEntropyLoss(reduction='none')
    dice_loss = losses.DiceLoss(n_classes=args.num_classes)

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    patience_performance = 0

    iterator = tqdm(range(max_epoch), initial=iter_num // len(trainloader), total=max_epoch, ncols=70)

    if model_sam is not None and args.sam_fussion_type == 'time_sum':
        sam_sum_weight = cal_time_sum_weight(iter_num, max_iterations)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].squeeze(), sampled_batch['label']
            volume_batch_strong = sampled_batch['image_strong'].squeeze()

            img_w = volume_batch[:args.labeled_bs]
            lab_w_s = label_batch[:args.labeled_bs]
            img_s = volume_batch_strong[:args.labeled_bs]

            uimg_w = volume_batch[args.labeled_bs:]
            uimg_s = volume_batch_strong[args.labeled_bs:]

            if args.use_mt != 'no':
                if model_sam is None:
                    out_ef_sams = None
                    sam_sum_weight = None
                else:
                    out_ef_sams = cal_sam_img_encoder(model_sam, uimg_w.unsqueeze(1))
                ulab_w_s = generate_ulab_by_teacher_model(uimg_w, ema_model, args.min_pixel_num_obj, num_classes, out_ef_sams, sam_sum_weight)
                ema_model.train()
            else:
                if model_sam is None:
                    out_ef_sams = None
                    sam_sum_weight = None
                else:
                    out_ef_sams = cal_sam_img_encoder(model_sam, uimg_w.unsqueeze(1))
                ulab_w_s = generate_ulab_by_model(uimg_w, model1, model2, args.min_pixel_num_obj, num_classes, out_ef_sams, sam_sum_weight)

            if model_sam is not None:
                del out_ef_sams
                torch.cuda.empty_cache()

            model1.train()
            model2.train()

            forward_mask_w_s = lab_w_s > 0
            forward_umask_w_s = ulab_w_s > 0

            new_labs = []
            new_imgs = []
            loss_mask_weights = []

            for i in range(len(img_w)):
                new_img, new_lab, loss_mask_weight = create_new_img_lab(imgs=torch.cat((img_w[i].unsqueeze(0), img_s[i].unsqueeze(0), uimg_w[i].unsqueeze(0), uimg_s[i].unsqueeze(0)), dim=0), labs=torch.cat((lab_w_s[i].unsqueeze(0), lab_w_s[i].unsqueeze(0), ulab_w_s[i].unsqueeze(0), ulab_w_s[i].unsqueeze(0)), dim=0), fms=torch.cat((forward_mask_w_s[i].unsqueeze(0), forward_mask_w_s[i].unsqueeze(0),forward_umask_w_s[i].unsqueeze(0),forward_umask_w_s[i].unsqueeze(0)),dim=0), u_weight=args.u_weight, min_total_pixel_num=args.min_total_pixel_num, is_self=True, crop_method=args.crop_method)
                new_imgs.append(new_img)
                new_labs.append(new_lab)
                loss_mask_weights.append(loss_mask_weight)

            new_imgs = torch.concat(new_imgs, dim=0)
            new_labs = torch.concat(new_labs, dim=0)
            loss_mask_weights = torch.concat(loss_mask_weights, dim=0)

            if model_sam is not None:
                out_ef_sams = cal_sam_img_encoder(model_sam, new_imgs)

            new_imgs, new_labs, loss_mask_weights = new_imgs.cuda(), new_labs.cuda(), loss_mask_weights.cuda()

            if  model_sam is None:
                out1_predicts = model1(new_imgs)  # unet
                out2_predicts = model2(new_imgs)  # resunet
                del new_imgs
            else:
                out1_predicts = model1(new_imgs, out_ef_sams, sam_sum_weight)  # unet
                out2_predicts = model2(new_imgs, out_ef_sams, sam_sum_weight)  # resunet
                del new_imgs, out_ef_sams
            torch.cuda.empty_cache()

            supervised_prob_loss1 = ((ce_loss(out1_predicts, new_labs.long()) * loss_mask_weights).mean()) + dice_loss(torch.softmax(out1_predicts, dim=1), new_labs.unsqueeze(1), mask=loss_mask_weights.unsqueeze(1))
            if args.use_cross_loss:
                cross_unsuper_prob_loss1 = ce_loss(out1_predicts, torch.argmax(F.softmax(out2_predicts.detach(), dim=1), dim=1).long()).mean() + dice_loss(torch.softmax(out1_predicts, dim=1), torch.argmax(F.softmax(out2_predicts.detach(), dim=1), dim=1).unsqueeze(1)).mean()

            if args.use_cross_loss == False:
                loss1 = supervised_prob_loss1
            else:
                loss1 = 2 * supervised_prob_loss1 + cross_unsuper_prob_loss1

            supervised_prob_loss2 = ((ce_loss(out2_predicts, new_labs.long()) * loss_mask_weights).mean()) + dice_loss(torch.softmax(out2_predicts, dim=1), new_labs.unsqueeze(1), mask=loss_mask_weights.unsqueeze(1))
            if args.use_cross_loss:
                cross_unsuper_prob_loss2 = ce_loss(out2_predicts, torch.argmax(F.softmax(out1_predicts.detach(), dim=1),dim=1).long()).mean() + dice_loss(torch.softmax(out2_predicts, dim=1), torch.argmax(F.softmax(out1_predicts.detach(), dim=1), dim=1).unsqueeze(1)).mean()

            if args.use_cross_loss == False:
                loss2 = supervised_prob_loss2
            else:
                loss2 = 2 * supervised_prob_loss2 + cross_unsuper_prob_loss2

            if args.use_union_loss:
                loss = loss1 + loss2
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
            else:
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

            if args.use_mt == 'unet':
                update_model_ema(model1, ema_model, 0.99)
            elif args.use_mt == 'res_unet':
                update_model_ema(model2, ema_model, 0.99)

            iter_num = iter_num + 1

            if model_sam is not None and args.sam_fussion_type == 'time_sum':
                sam_sum_weight = cal_time_sum_weight(iter_num, max_iterations)

            lr_ = args.base_lr * (1.0 - (iter_num + start_iterations) / total_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            if args.use_union_loss:
                logging.info('iteration: %d, total_loss: %f, loss_unet: %f, loss_resunet: %f' % (iter_num, loss, loss1, loss2))
            else:
                logging.info('iteration: %d, loss_unet: %f, loss_resunet: %f' % (iter_num, loss1, loss2))

            if (iter_num > 0 and iter_num % 9000 == 0) or (iter_num == max_iterations-1):
                model1.eval()
                model2.eval()
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i_1 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model1, patch_size=args.image_size, classes=num_classes)
                    metric_i_2 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model2, patch_size=args.image_size, classes=num_classes)
                    metric_list_1 += np.array(metric_i_1)
                    metric_list_2 += np.array(metric_i_2)

                metric_list_1 = metric_list_1 / len(db_val)
                metric_list_2 = metric_list_2 / len(db_val)

                performance1 = np.mean(metric_list_1, axis=0)
                performance2 = np.mean(metric_list_2, axis=0)

                if args.use_early_stop:
                    if performance1 < best_performance1 and performance2 < best_performance2:
                        patience_performance += 1
                        if patience_performance == 2:
                            break

                if performance1 > best_performance1:
                    patience_performance = 0
                    best_performance1 = performance1
                    save_best_path = os.path.join(ad_path, 'best_model_unet.pth')
                    save_net_opt(model1, optimizer1, save_best_path)

                if performance2 > best_performance2:
                    patience_performance = 0
                    best_performance2 = performance2
                    save_best_path = os.path.join(ad_path, 'best_model_resunet.pth')
                    save_net_opt(model2, optimizer2, save_best_path)

                model1.train()
                model2.train()





