"""Train a tiny classifier on CPU/CUDA and export and verify embedded C99.

Synthetic data uses separate train, validation, and held-out test splits.
Results demonstrate the workflow rather than an application benchmark.
"""

from neura_command import (
    Sequential,
    Dense,
    BatchNorm,
    ReLU,
    Softmax,
    Adam,
    CrossEntropyLoss,
    EarlyStopping,
    set_random_seed,
)
from demo_utils import (
    demo_parser,
    parse_demo,
    classification_data,
    split_data,
    save_data,
    export_and_measure,
)


def main(argv=None):
    parser = demo_parser(__doc__, output_dir="build/embedded-demo")
    args = parse_demo(parser, argv)
    try:
        set_random_seed(args.seed, args.device)
        splits = split_data(*classification_data(args.seed, args.samples))
        save_data(args.output_dir, splits)
        model = Sequential(
            [
                Dense(6, args.hidden),
                BatchNorm(args.hidden),
                ReLU(),
                Dense(args.hidden, 3),
                Softmax(),
            ],
            optimizer=Adam(learning_rate=0.02),
            loss=CrossEntropyLoss(),
            device=args.device,
        )
        history = model.train(
            *splits[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=splits[1],
            callbacks=[EarlyStopping(patience=12)],
            verbose=False,
            seed=args.seed,
        )
        return export_and_measure(model, history, args, splits[2], "classification")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
