"""Train a compact two-output sensor regressor and compare float32, INT8, and C99."""

from neura_command import (
    Sequential,
    Dense,
    Tanh,
    Adam,
    MSELoss,
    EarlyStopping,
    set_random_seed,
)
from demo_utils import (
    demo_parser,
    parse_demo,
    regression_data,
    split_data,
    save_data,
    export_and_measure,
)


def main(argv=None):
    parser = demo_parser(
        __doc__, output_dir="build/regression-demo", epochs=100, samples=400
    )
    args = parse_demo(parser, argv)
    try:
        set_random_seed(args.seed, args.device)
        splits = split_data(*regression_data(args.seed, args.samples))
        save_data(args.output_dir, splits)
        model = Sequential(
            [
                Dense(4, args.hidden, weight_init="xavier"),
                Tanh(),
                Dense(args.hidden, 2),
            ],
            optimizer=Adam(learning_rate=0.01),
            loss=MSELoss(),
            device=args.device,
        )
        history = model.train(
            *splits[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=splits[1],
            callbacks=[EarlyStopping(patience=15)],
            verbose=False,
            seed=args.seed,
        )
        return export_and_measure(model, history, args, splits[2], "regression")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
