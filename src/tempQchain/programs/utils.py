import torch
from domiknows.graph.dataNode import DataNode


def to_int_list(x):
    return torch.LongTensor([int(i) for i in x])


def to_float_list(x):
    return torch.Tensor([float(i) for i in x])


def str_to_int_list(x) -> torch.LongTensor:
    return torch.LongTensor([int(i) for i in x])


def read_label(_, label):
    return label


def check_symmetric(arg1: DataNode, arg2: DataNode) -> bool:
    if arg1 == arg2:
        return False
    relation_arg2 = arg2.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(",")
    if relation_describe[0] == "symmetric":
        qid1 = arg1.getAttribute("id").item()
        if qid1 == int(relation_describe[1]):
            return True
    return False


def check_transitive(arg11: DataNode, arg22: DataNode, arg33: DataNode) -> bool:
    if arg11 == arg22 or arg11 == arg33 or arg22 == arg33:
        return False
    relation_arg3 = arg33.getAttribute("relation")
    if relation_arg3 == "":
        return False
    # exmple of relation_describe: ['transitive', '1', '4']
    relation_describe = relation_arg3.split(",")

    if relation_describe[0] == "transitive":
        qid1 = arg11.getAttribute("id").item()
        qid2 = arg22.getAttribute("id").item()
        if qid1 == int(relation_describe[1]) and qid2 == int(relation_describe[2]):
            return True
    return False
