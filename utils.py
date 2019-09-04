import csv
from tensorflow.python import pywrap_tensorflow
from collections import namedtuple

LabelData = namedtuple("LabelData",
                       ["name", "multi_dim", "probabilities", "num_classes"])


def build_label_list_from_file(label_file):
    total_classes = 0
    label_list = []
    with open(label_file) as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            new_label = LabelData(name=row[0],
                                  multi_dim=row[1] == 'multi_dim',
                                  probabilities=[float(v) for v in row[2:]],
                                  num_classes=len([float(v) for v in row[2:]]))
            label_list.append(new_label)
            if row[1] != 'one_dim' and row[1] != 'multi_dim':
                raise ValueError("label type must be one_dim or multi_dim, is %s" % row[1])

            total_classes += 1 if not new_label.multi_dim else new_label.num_classes
    return label_list, total_classes


def filter_vars_with_checkpoint(chkpt_path, var_list):
    """
    :param chkpt_path: path to checkpoint file
    :param var_list: list of variables to filter
    :return: list of var_list members contained in checkpoint
    """
    reader = pywrap_tensorflow.NewCheckpointReader(chkpt_path)
    var_names = reader.get_variable_to_shape_map()

    filtered_vars = []
    for v in var_list:
        name = v.name[0:v.name.index(":")]
        if name in var_names:
            filtered_vars.append(v)
    return filtered_vars
