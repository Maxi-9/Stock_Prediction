import click

from TimeSeriesPrediction.model import Commons


class Parse_Args:
    def __init__(self):
        pass

    filename = click.argument("filename", nargs=1, type=click.Path(exists=False))
    modeltype = click.argument(
        "mtype", nargs=1, type=click.Choice(Commons.model_mapping.keys())
    )

    save_xlsx = click.option(
        "-x",
        "--save",
        type=str,
        default=None,
        help="Saves test table to specified file as xlsx file, used for debugging and testing.",
    )
    debug = click.option(
        "-d",
        "--debug",
        is_flag=True,
        help="Prints more info: prints debug table and more metrics.",
    )

    overwrite = click.option(
        "-o",
        "--overwrite",
        is_flag=True,
        help="Overwrites (if exists) else trains pre-existing model.",
    )

    cache = click.option(
        "-c",
        "--cache",
        type=str,
        default="cache",
        help="Creates a cache file for each stock in provided directory, overwrites the rows that already exist."
    )

    log = click.option(
        "-l",
        "--log",
        type=str,
        default="log",
        help="Creates a log file for each stock, designed as a way to measure the performance of the models in a with real world scenario."
    )

    @staticmethod
    def stocks(default=None, multiple=True):
        if default is None:
            default = []
        req = default is None
        return click.option(
            "-s",
            "--stocks",
            multiple=multiple,
            type=str,
            default=default,
            required=req,
            help="""\b
        Stocks should be in on of the following forms:
        - stock
        - stock.market
        To specify data period, start/stop date, or just a start date(to current date), use:
        - stock,period     | With period in form of: 1d, 1mo, 1y ytd, max
        - stock:start,stop | Start and stop being date in form: MM-DD-YYYY
        - stock:start      | Start being date in form: MM-DD-YYYY\n
        """,
        )

    seed = click.option(
        "-e",
        "--seed",
        type=int,
        default=None,
        help="""\b
    Random seed if not specified. Set fixed seed for supported models, setting seed will make the model deterministic but the input data from yFinance isn't deterministic.\n""",
    )  # Warning: May not work for all models, if you create your own model, you customize it to set seed

    split = click.option(
        "-t",
        "--split",
        type=float,
        default=0.8,
        help="Splits training and test data. Higher value means more training data (Input a float between 0 and 1).",
    )

    @staticmethod
    def parser(help_text: str):
        return click.command(help=help_text)
