"""Run-id naming."""


def make_run_id(model: str, depth: int, connection: str, n: int,
                dataset: str, fraction: int, seed: int) -> str:
    """`{model}{depth}_{connection}_n{n}_{dataset}p{fraction}_s{seed}`.

    `fraction` is the integer percent of the training set (100, 25, 10).
    Example: resnet56_dhc_n4_cifar100p10_s1.
    """
    return f"{model}{depth}_{connection}_n{n}_{dataset}p{fraction}_s{seed}"


def run_id_of(cfg: dict) -> str:
    """The run id of a resolved config."""
    m, c, d = cfg["model"], cfg["connection"], cfg["data"]
    return make_run_id(m["name"], m["depth"], c["kind"], c["n"], d["name"], d["percent"], cfg["seed"])
