import argparse
import datetime
import os
import train, evaluation
import glob
import numpy as np
import shutil
import json
import tensorflow as tf

def copy_source_to_model_dir(model_dir):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    script_files = [fname for fname in os.listdir(script_dir) if fname[-3:] == ".py"]
    target_dir = os.path.join(model_dir, "code")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for fname in script_files:
        shutil.copyfile(os.path.join(script_dir, fname), os.path.join(target_dir, fname))


def write_load_script(write_dir, args, script_name, switches, *parser_groups, print_result=False):
    """
    Writes .bat and .sh files to automatically perform an action based on the arguments for this
    run.  For example, resuming with the same arguments, or sampling with the same architecture
    Uses parser groups to determine which arguments to include
    :param model_dir: directory to place this script (which includes "code" directory with
        the code for this model)
    :param args: dictionary of command line arguments for this run
    :param script_name: name of the script to write
    :param switches: extra switches to add
    :param parser_groups: parser groups to include (for example, sampling doesn't need training groups)
    :param print_result: print the script parameters
    :return:
    """
    run_script = os.path.join(write_dir, "code", "run.py")
    parameters = switches # + ["\t--model_dir \"%s\"" % args.model_dir]
    action_list = []
    for group in parser_groups:
        action_list += group._group_actions

    for action in action_list:
        switch = action.option_strings[0]
        name = action.dest
        default = action.default
        if args[name] != default:
            if isinstance(args[name], bool) and args[name]:
                parameters.append("\t%s" % switch)
            elif isinstance(args[name], str):
                parameters.append("\t%s \"%s\"" % (switch, args[name]))
            else:
                parameters.append("\t%s %s" % (switch, args[name]))

    with open(os.path.join(model_dir, "%s_script.sh" % script_name), "w") as f:
        bash_parameters = " \\\n".join(parameters)
        f.write("python %s \\\n%s" % (run_script, bash_parameters))
    with open(os.path.join(model_dir, "%s_script.bat" % script_name), "w") as f:
        cmd_parameters = " ^\n".join(parameters)
        f.write("python %s ^\n%s" % (run_script, cmd_parameters))
    if print_result:
        print("Options %s" % script_name)
        print("\\\n".join(parameters))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters for stylegan training")

    action_group = parser.add_mutually_exclusive_group(required=True)

    action_group.add_argument("--resume", action='store_true', default=False,
                              help="resume from latest checkpoint in the log dir")
    action_group.add_argument("--sample", action='store_true', default=False,
                              help="sample instead of training")
    action_group.add_argument("--train", action='store_true', default=False,
                              help="generate a subdirectory (in model_dir) to use as the model directory")

    eval_group = parser.add_argument_group("--eval_group")
    eval_group.add_argument("--grid", action='store_true', default=False,
                             help="sample a grid")
    eval_group.add_argument("--mix_layer", type=int, default=None,
                             help="layer to style mix at")
    eval_group.add_argument("--save_sample_graph", action='store_true', default=False,
                             help="save sampler graph")
    eval_group.add_argument("--psi_w", type=float, default=1.,
                             help="psi for truncation trick (page 8)")
    eval_group.add_argument("--num_samples", type=int, default=128,
                             help="total samples to generate")
    logging_and_debugging_group = parser.add_argument_group("logging_and_debugging")
    logging_and_debugging_group.add_argument("--eager", action='store_true', default=False,
                                             help="use eager mode")
    logging_and_debugging_group.add_argument("--beholder", action='store_true', default=False,
                                             help="use beholder logging")
    logging_and_debugging_group.add_argument("--tboard_debug", action='store_true', default=False,
                                             help="wrap session with tensorboard debugger")
    logging_and_debugging_group.add_argument("--cli_debug", action='store_true', default=False,
                                             help="wrap session with cli debugger")
    logging_and_debugging_group.add_argument("--profile", action='store_true', default=False,
                                             help="generate profile data")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--res_h", type=int, default=32)
    model_group.add_argument("--res_w", type=int, default=32)
    model_group.add_argument("--start_res_h", type=int, default=4,
                             help="starting resolution (by height)")
    model_group.add_argument("--start_res_w", type=int, default=4,
                             help="starting resolution (by width)")
    model_group.add_argument("--conditional_type", type=str, default=None,
                             help="[proj|acgan]")
    model_group.add_argument("--map_cond", action='store_true', default=False,
                             help="input conditional variables to mapping network,"
                             " otherwise will be added to intermeidate")
    model_group.add_argument("--cond_layers", type=str, default=None,
                             help="comma separated list of layers to apply conditioning to, defaults to all")
    model_group.add_argument("--cond_weight", type=float, default=0.5,
                             help="multiply conditional portion by this")
    model_group.add_argument("--label_file", type=str, default=None,
                              help="use cgan discriminator "
                             "file is a csv with each line giving label info "
                             "format is: name, possible value 1, ratio of value 1, etc "
                             "for example: sun_glasses, 1., .1, 0., .9 to indicate 1. means "
                             "sun_glasses present, which happens 10% of the time, 0. indicates no sun_glasses, "
                             "which happens 90% of the time")

    model_group.add_argument("--pixel_norm", action='store_true', default=False,
                             help="do pixel norm")
    model_group.add_argument("--resize_method", type=str, default="bilinear", required=False,
                             help="method used to up/downsample images")
    model_group.add_argument("--traditional_input", action='store_true', default=False,
                             help="use traditional input to generator")
    model_group.add_argument("--no_mapping_network", action='store_true', default=False,
                             help="don't use a mapping network")
    model_group.add_argument("--no_equalized_lr", action='store_true', default=False,
                             help="don't use equalized learning rate")
    model_group.add_argument("--no_noise_added", action='store_true', default=False,
                             help="don't add noise each layer")
    model_group.add_argument("--no_minibatch_stddev", action='store_true', default=False,
                             help="don't apply minibatch stddev to last discriminator layer")

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--loss_fn", type=str, default="wgan",
                                help="loss function [non_saturating|wgan]")
    training_group.add_argument("--gp", type=str, default=None,
                                help="gradient penalty [wgan|r1]")
    training_group.add_argument("--lambda_gp", type=float, default=10.,
                                help="lambda for gradient penalty")
    training_group.add_argument("--lambda_drift", type=float, default=.001,
                                help="lambda for drift penalty: penalizes high output values")
    training_group.add_argument("--ncritic", type=int, default=1,
                                help="# discriminator runs per generator run")
    training_group.add_argument("--steps_per_save", type=int, default=None,
                                help="# steps between saving the model, defaults to end of phase")
    training_group.add_argument("--no_train", action='store_true', default=False,
                                help="don't apply gradient updates")
    training_group.add_argument("--batch_size", type=str, default="16",
                                help="minibatch size "
                                     "(can schedule with dict: {4: 32, 8: 16}"
                                     " means 32 for res 4, 16 for res 8")
    training_group.add_argument("--epochs_per_res", type=int, default=5,
                                help="epochs to run per resolution")
    training_group.add_argument("--learning_rate", type=float, default=.001,
                                help="initial learning rate")
    training_group.add_argument("--optimizer", type=str, default="adam",
                                help="optimizer [adam|gradient_descent]")
    training_group.add_argument("--adam_beta1", type=float, default=0.,
                                help="beta1, if None will use default")
    training_group.add_argument("--adam_beta2", type=float, default=.99,
                                help="beta2, if None will use default")
    training_group.add_argument("--decay_learning_rate", action='store_true',
                                default=False, help="should decay learning rate [UNIMPLEMENTED]")
    training_group.add_argument("--cuda_visible_devices", type=str, default=None,
                                help="set CUDA_VISIBLE_DEVICES")
    training_group.add_argument("--input_file_regex", type=str,
                                help="path with regex for input files (ex: data/*.jpg)")
    training_group.add_argument("--cond_uniform_fake", action='store_true', default=False,
                                help="when generating fake images for training, use uniform distribution"
                                     "for conditionals instead of marginal dist from real data")

    dir_group = parser.add_argument_group("directories")
    dir_group.add_argument("--model_dir", type=str, required=False,
                           help="directory to store/load models and logs (will resume if possible)")


    np.set_printoptions(suppress=True)  # prevents scientific notation prints
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    else:
        if args.sample:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "\"\""
    if args.train:
        if 'EXPERIMENT_TITLE' in os.environ:
            title = os.environ['EXPERIMENT_TITLE']
        else:
            title = input("Title (can set with environ variable EXPERIMENT_TITLE): ")
        if title == "":
            title = "TESTING"
        print("Title: %s" % title)
        generated_dir = title+"_stylegan_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join(
            args.model_dir,
            generated_dir)
        os.makedirs(model_dir)
        # running this script will generate a sample from the current model
        arg_dict = vars(args)
        write_load_script(model_dir, vars(args), "train", ["\t--train"], model_group, training_group,
                          logging_and_debugging_group, dir_group, print_result=True)
        arg_dict['model_dir'] = model_dir
        write_load_script(model_dir, vars(args), "sample", ["\t--sample",
                                                            "\t--psi_w .25",
                                                            "\t--batch_size 32"], model_group, dir_group)
        # running this script will resume training at the latest resolution
        write_load_script(model_dir, vars(args), "resume", ["\t--resume"], model_group, training_group, dir_group)
        copy_source_to_model_dir(model_dir)
    else:
        model_dir = args.model_dir

    if args.loss_fn == "non_saturating":
        loss_fn = train.non_saturating_loss
    elif args.loss_fn == "wgan":
        loss_fn = train.wasserstein_loss
    else:
        raise ValueError("Unknown loss function %s" % args.loss_fn)
    if args.gp == "wgan":
        if args.eager:
            gp_fn = train.wgan_gp_eager
        else:
            gp_fn = train.wgan_gp
    elif args.gp == "r1":
        gp_fn = train.r1_gp
    elif args.gp is None:
        gp_fn = None
    else:
        raise ValueError("Unknown gradient penalty %s" % args.gp)
    if args.ncritic < 1:
        raise ValueError("ncritic must be at least 1")

    if args.cond_layers is not None:
        cond_layers = [int(v) for v in args.cond_layers.split(",")]
    else:
        cond_layers = None

    save_paths = train.SavePaths(
        gen_model=os.path.join(model_dir, "saved_gen.h5"),
        dis_model=os.path.join(model_dir, "saved_dis.h5"),
        mapping_network=os.path.join(model_dir, "saved_mn.h5"),
        sampling_model=os.path.join(model_dir, "saved_sampler.h5"),
        gen_optim=os.path.join(model_dir, 'g_optim', 'g_optim'),
        dis_optim=os.path.join(model_dir, 'd_optim', 'd_optim'),
        mn_optim=os.path.join(model_dir, 'mn_optim', 'mn_optim'),
        alpha=os.path.join(model_dir, "alpha.txt"),
        step=os.path.join(model_dir, "step.txt")
    )

    # "summary_[num]" directories contain summaries for resolution [num]
    # the highest [num] indicates the most recent resolution
    existing_summary_dirs = glob.glob(os.path.expanduser(os.path.join(model_dir, "summary_*")))
    if len(existing_summary_dirs) > 0:
        summary_dir_res = [int(os.path.basename(fname)[len("summary_"):]) for fname in existing_summary_dirs]
        highest_res = sorted(summary_dir_res)[-1]
    else:
        if args.start_res_h != args.start_res_w:
            highest_res = max(args.start_res_w, 4)  # default starting value
        else:
            highest_res = max(args.start_res_w, 8)  # default starting value

    if args.input_file_regex is not None:
        files = glob.glob(os.path.expanduser(args.input_file_regex))

    hps = train.TrainHps(
        res_h=args.res_h,
        res_w=args.res_w,
        current_res_w=highest_res,
        batch_size=int(args.batch_size) if "," not in args.batch_size else args.batch_size,
        epochs_per_res=args.epochs_per_res,
        psi_w=args.psi_w,
        optimizer=args.optimizer,
        loss_fn=loss_fn,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        model_dir=model_dir,
        gp_fn=gp_fn,
        lambda_gp=args.lambda_gp,
        lambda_drift=args.lambda_drift,
        ncritic=args.ncritic,
        do_pixel_norm=args.pixel_norm,
        do_equalized_lr=not args.no_equalized_lr,
        do_minibatch_stddev=not args.no_minibatch_stddev,
        do_traditional_input=args.traditional_input,
        do_mapping_network=not args.no_mapping_network,
        do_add_noise=not args.no_noise_added,
        resize_method=args.resize_method,
        start_res_h=args.start_res_h,
        start_res_w=args.start_res_w,
        cli_debug=args.cli_debug,
        tboard_debug=args.tboard_debug,
        eager=args.eager,
        no_train=args.no_train,
        steps_per_save=args.steps_per_save,
        save_paths=save_paths,
        profile=args.profile,
        label_file=args.label_file,
        conditional_type=args.conditional_type,
        cond_weight=args.cond_weight,
        cond_layers=cond_layers,
        map_cond=args.map_cond,
        cond_uniform_fake=args.cond_uniform_fake,
        use_beholder=args.beholder)
    #print("Hyperparameters:")
    #print(pprint(dict(vars(hps))))
    print("Model dir: %s" % model_dir)
    with open(os.path.join(model_dir, "hps.txt"), "w") as f:
        json.dump(hps._asdict(), f, indent=1, default=lambda x: 'function')

    if args.label_file is not None and args.train:
        try:
            shutil.copyfile(args.label_file, os.path.join(model_dir, os.path.basename(args.label_file)))
        except shutil.SameFileError:
            pass

    if args.sample:
        sample_dir = os.path.join(hps.model_dir, "samples")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        if args.save_sample_graph:
            saved_model_dir = os.path.join(hps.model_dir, "saved_graph")
            evaluation.save_sampling_graph(hps, save_paths, saved_model_dir)
        elif args.grid:
            evaluation.sample_grid(hps, sample_dir)
        elif args.mix_layer is not None:
            evaluation.sample_multiple_mix(hps, sample_dir, args.mix_layer, args.num_samples)
        else:
            evaluation.sample_multiple(hps, sample_dir, args.num_samples)
    else:
        train.train(hps, files)
