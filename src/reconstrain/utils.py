from pytorch_lightning.utilities.argparse import get_init_arguments_and_types


def add_args_from_init(cls, group):
    args = get_init_arguments_and_types(cls)
    for name, types, default in args:
        if types[0] not in (int, float, str, bool):
            continue
        group.add_argument(f"--{name}", type=types[0], default=default)
    return group


def auto_args(target):
    @classmethod
    def add_args(cls, group):
        for base in cls.__bases__:
            if hasattr(base, "add_args"):
                group = base.add_args(group)
        return add_args_from_init(cls, group)

    setattr(target, "add_args", add_args)
    return target
