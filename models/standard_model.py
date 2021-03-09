import torch
import torch.nn as nn
import copy


class StandardModel(nn.Module):
    """
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    """
    def __init__(self, net):
        super(StandardModel, self).__init__()
        # init victim model
        self.net = net

        # init victim model meta-information
        if len(self.net.input_size) == 3:
            self.mean = torch.FloatTensor(self.net.mean).view(1, -1, 1, 1)
            self.std = torch.FloatTensor(self.net.std).view(1, -1, 1, 1)
        else:
            # must be debug dataset
            assert len(self.net.input_size) == 1
            self.mean = torch.FloatTensor(self.net.mean).view(1, -1)
            self.std = torch.FloatTensor(self.net.std).view(1, -1)
        self.input_space = self.net.input_space  # 'RGB' or 'GBR' or 'GRAY'
        self.input_range = self.net.input_range  # [0, 1] or [0, 255]
        self.input_size = self.net.input_size

    def whiten(self, x):
        # channel order
        if self.input_space == 'BGR':
            x = x[:, [2, 1, 0], :, :]  # pytorch does not support negative stride index (::-1) yet

        # input range
        x = torch.clamp(x, 0, 1)
        if max(self.input_range) == 255:
            x = x * 255

        # normalization
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        if self.std.device != x.device:
            self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std

        return x

    def forward(self, x):
        raise NotImplementedError


class StandardVictimModel(StandardModel):
    """
    This model inherits StandardModel class, and maintain a record for query counts and best adversarial examples
    """

    def __init__(self, net):
        super(StandardVictimModel, self).__init__(net=net)

        # init attack states
        self.image = None
        self.attack_type = None
        self.norm_type = None
        self.label = None
        self.target_label = None
        self.query_count = 0
        self.last_adv_image = None
        self.last_adv_label = None
        self.last_success = None
        self.last_distance = None
        self.best_adv_image = None
        self.best_adv_label = None
        self.best_distance = None
        # self.best_success = None  # best adv image is always and success attack thus could be omitted

    def forward(self, x, no_count=False):
        if not no_count:
            # increase query counter
            self.query_count += x.shape[0]

        # whiten input
        x = self.whiten(x)

        # forward
        x = self.net(x)
        return x

    def reset(self, image, label, target_label, attack_type, norm_type):
        self.image = image.clone()
        self.attack_type = attack_type
        self.norm_type = norm_type
        if self.attack_type == 'untargeted':
            assert label is not None
            self.label = label.clone().view([])
        elif self.attack_type == 'targeted':
            assert target_label is not None
            self.target_label = target_label.clone().view([])
        else:
            raise NotImplementedError('Unknown attack type: {}'.format(self.attack_type))

        self.query_count = 0
        self.last_adv_image = None
        self.last_adv_label = None
        self.last_distance = None
        self.last_success = None
        self.best_adv_image = None
        self.best_adv_label = None
        self.best_distance = None

    def calc_mse(self, adv_image):
        assert self.image is not None
        diff = adv_image - self.image
        diff = diff.view(diff.shape[0], -1)
        return (diff ** 2).sum(dim=1) / self.image.numel()

    def calc_distance(self, adv_image):
        assert self.image is not None
        diff = adv_image - self.image
        diff = diff.view(diff.shape[0], -1)
        if self.norm_type == 'l2':
            return torch.sqrt((diff ** 2).sum(dim=1))
        elif self.norm_type == 'linf':
            return diff.abs().max(dim=1)[0]
        else:
            raise NotImplementedError('Unknown norm: {}'.format(self.norm_type))

    def _is_valid_adv_pred(self, pred):
        if self.attack_type == 'untargeted':
            return ~(pred.eq(self.label))
        elif self.attack_type == 'targeted':
            return pred.eq(self.target_label)
        else:
            raise NotImplementedError('Unknown attack type: {}'.format(self.attack_type))

    def query(self, adv_image, sync_best=True, no_count=False):
        adv_image = adv_image.detach()
        assert self.attack_type is not None
        pred = self.forward(adv_image, no_count=no_count).argmax(dim=1)
        distance = self.calc_distance(adv_image)
        success = self._is_valid_adv_pred(pred)

        # check if better adversarial examples are found
        if sync_best:
            if success.any().item():
                if self.best_distance is None or self.best_distance.item() > distance[success].min().item():
                    # if better adversarial examples are found, record it
                    adv_index = distance[success].argmin()
                    best_adv_image = adv_image[success][adv_index].view(self.image.shape)
                    best_adv_label = pred[success][adv_index].view([])
                    best_distance = distance[success][adv_index].view([])
                    failed = False

                    # Since cuda/cudnn is extensively optimized for batch-ed inputs, we might get slightly different
                    # results for the following two scenarios:
                    # 1. wrap example X and some other examples into a batch, forward and see logit for X
                    # 2. forward example X only (i.e., batch size = 1), and see logit for X
                    # As a result, if logit(label) is too close to logit(target), there might be some numerical problems
                    # Since our program might think the predicted class as the label class sometimes, and think it as
                    # the target class for other times, which will definitely destroy the assumption for binary search
                    # So, here we try to fix this by adding a small vector to adv_image if this happens, and we
                    # explicitly exclude these queries from total counts since this phenomenon is mainly caused by
                    # unsatisfied computing infrastructure instead of intrinsic requirement of the attacking algorithm
                    # And this problem could possibly been solved by future updates of gpus or cuda/cudnn
                    if not self._is_valid_adv_pred(self.forward(best_adv_image, no_count=True).argmax()).item():
                        best_adv_image = self.image + (1 + 1e-6) * (best_adv_image - self.image)
                        best_adv_label = self.forward(best_adv_image, no_count=True).argmax().view([])
                        best_distance = self.calc_distance(best_adv_image).view([])
                        if not self._is_valid_adv_pred(best_adv_label).item():
                            # cannot fix numerical problem after adding a small factor of (adv_image - image)
                            failed = True

                    if (not failed) and \
                            (self.best_distance is None or self.best_distance.item() > best_distance.item()):
                        # if new adv image is still better than previous after fixing, we should record it
                        self.best_adv_image = best_adv_image
                        self.best_adv_label = best_adv_label
                        self.best_distance = best_distance
                    else:
                        # or else we discard it
                        pass

            self.last_adv_image = adv_image.clone()
            self.last_adv_label = pred
            self.last_distance = distance
            self.last_success = success
        return success


class StandardPolicyModel(StandardModel):
    """
    This model inherits StandardModel class
    """

    def __init__(self, net):
        super(StandardPolicyModel, self).__init__(net=net)
        self.init_state_dict = copy.deepcopy(self.state_dict())
        self.factor = 1.0

        # for policy models, we do whiten in policy.net.forward() instead of policy.forward()
        # since _inv models requires grad w.r.t. input in range [0, 1]
        self.net.whiten_func = self.whiten

    def forward(self, adv_image, image=None, label=None, target=None,
                output_fields=('grad', 'std', 'adv_logit', 'logit')):
        # get distribution mean, (other fields such as adv_logit and logit will also be in output)
        output = self.net(adv_image, image, label, target, output_fields)

        # we have two solutions for scaling:
        # 1. multiply scale factor into mean and keep std unchanged
        # 2. keep mean unchanged and make std divided by scale factor

        # since we only optimize mean (if args.exclude_std is True) and we often use some form of momentum (SGDM/Adam),
        # changing the scale of mean will change the scale of gradient and previous momentum may no longer be suitable
        # so we choose solution 2 for std here: std <-- std / self.factor
        if 'std' in output:
            output['std'] = output['std'] / self.factor

        # only return fields requested, since DistributedDataParallel throw error if unnecessary fields are returned
        return {field_key: output[field_key] for field_key in output_fields if field_key in output}

    def reinit(self):
        self.load_state_dict(self.init_state_dict)
        self.factor = 1.0

    def rescale(self, scale):
        self.factor *= scale
