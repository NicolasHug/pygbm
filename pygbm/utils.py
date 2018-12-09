from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

from .gradient_boosting import GradientBoostingClassifier


def get_lightgbm_estimator(pygbm_est):
    """Return an estimator from LightGBM that has the same hyperparameters
    as pygbm_est"""

    pygbm_params = pygbm_est.get_params()

    if pygbm_params['loss'] == 'auto':
        raise ValueError('auto loss is not accepted. We need to know if '
                         'the problem is binary or multiclass classification.')
    if pygbm_params['scoring'] is not None:
        raise ValueError('Early stopping should be deactivated.')

    loss_mapping = {
        'least_squares': 'regression_l2',
        'binary_crossentropy': 'binary',
        'categorical_crossentropy': 'multiclass'
    }

    lgbm_params = {
        'objective': loss_mapping[pygbm_params['loss']],
        'learning_rate': pygbm_params['learning_rate'],
        'n_estimators': pygbm_params['max_iter'],
        'num_leaves': pygbm_params['max_leaf_nodes'],
        'max_depth': pygbm_params['max_depth'],
        'min_data_in_leaf': pygbm_params['min_samples_leaf'],
        'lambda_l2': pygbm_params['l2_regularization'],
        'max_bin': pygbm_params['max_bins'],
        'min_data_in_bin': 1,
        'min_sum_hessian_in_leaf': 1e-3,
        'min_gain_to_split': 0,
        'verbosity': 10 if pygbm_params['verbose'] else 0
    }
    # TODO: change hardcoded values when / if they're arguments to the
    # estimator.

    if pygbm_params['loss'] == 'categorical_crossentropy':
        # LGBM multiplies hessians by 2 in multiclass loss.
        lgbm_params['min_sum_hessian_in_leaf'] *= 2
        lgbm_params['learning_rate'] *= 2

    Est = (LGBMClassifier if isinstance(pygbm_est, GradientBoostingClassifier)
           else LGBMRegressor)

    return Est(**lgbm_params)
