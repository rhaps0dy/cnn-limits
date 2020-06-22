import sacred

import cnn_limits.sacred_utils as SU
from experiments import predict_cv_acc, save_new

experiment = sacred.Experiment("kernel_plus_learn", [
    SU.ingredient, save_new.experiment, predict_cv_acc.experiment])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.config
def _config():
    do_save = True
    do_loo = True
    do_4cv = True
    kmp_override = None

@experiment.automain
def main(do_save, do_loo, do_4cv, kmp_override):
    if do_save:
        save_new.main()

    if kmp_override is None:
        kmp_override = SU.base_dir()

    if do_loo:
        predict_cv_acc.main_no_eig(kernel_matrix_path=kmp_override, multiply_var=False, apply_relu=False, n_splits=-1)
    if do_4cv:
        predict_cv_acc.main_no_eig(kernel_matrix_path=kmp_override, multiply_var=False, apply_relu=False, n_splits=4)
