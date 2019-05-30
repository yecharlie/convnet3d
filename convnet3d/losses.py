import keras.backend as K
from . import backend


def detectionLossOHEM():
    def _ohemLoss(y_true, y_pred):
        cls_loss = K.sparse_categorical_crossentropy(y_true, y_pred)

        sorted_loss, sort_indices = backend.top_k(cls_loss, K.shape(cls_loss)[0])

        # hard samples have loss in the top 50%
        num_hard_samples = K.shape(cls_loss)[0] // 2

        # others are classified as easy samples
        num_easy_samples = K.shape(cls_loss)[0] - num_hard_samples

        # all hard samples are choosed
        hard_samples_indices = sort_indices[:num_hard_samples]

        # random choose a half of easy samples
        random_choices = num_hard_samples + backend.top_k(backend.uniform([num_easy_samples]), num_easy_samples // 2)[1]
        easy_samples_indices = K.gather(sort_indices, random_choices)
        total_choosed_indices = K.stop_gradient(K.concatenate([hard_samples_indices, easy_samples_indices]))
        choosed_samples_loss = K.gather(cls_loss, total_choosed_indices)
        ohem_loss = K.mean(choosed_samples_loss, axis=0)
        return ohem_loss
    return _ohemLoss


def reductionRegLoss():
    def _reductionRegLoss(y_true, y_pred):
        '''Compute the regression part of the hybrid loss.

        Args:
            y_true : Tensor from the generator of shape (B,5). The last value is the class of target. (0:negative, 1:positive)
            y_pred : Tensor from the network of shape (B,4).
        '''
        regression = y_pred
        regression_target = y_true[:, :-1]
        states = y_true[:, -1]

        # filter out negative samples
        indices = backend.where(K.greater(states, 0))[:, 0]
        regression = K.gather(regression, indices)
        regression_target = K.gather(regression_target, indices)
        regression_diff = K.abs(regression_target - regression)

        # smooth l1 with sigma = 1
        sigma = 1
        sigma_squared = sigma ** 2
        regression_loss = backend.where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of regressors
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, K.floatx())
        return K.sum(regression_loss) / normalizer
    return _reductionRegLoss
