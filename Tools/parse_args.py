import click

from model import Commons


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
        help="Saves test table to specified file as xlsx file.",
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

    # ignoreTestList = click.option(
    #     "-i",
    #     "--ignoreList",
    #     is_flag=True,
    #     help="Don't test to see if training stocks are on the test list from test.py.",
    # )

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
    Random seed if not specified. Set fixed seed for supported models.\n""",
    )  # Warning: May not work for all models

    split = click.option(
        "-t",
        "--split",
        type=float,
        default=0.8,
        help="Splits training and test data. Higher value means more training data (0-1 value).",
    )

    @staticmethod
    def parser(help_text: str):
        return click.command(help=help_text)
