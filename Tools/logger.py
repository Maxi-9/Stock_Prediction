# Time capsule logging of prediction for to ensure accuracy with real current data.
import csv
import os


class Logger:
    """
    Used for logging predicted data for the purposes of later analysis. Goal is to validate models and ensure no data leakage.
    The point being that it would be impossible for the model to know what the close value is because it hasn't happened yet.
    Format of log file:
    Date, {All columns of interest}, predicted value, buy signal
    """

    def __init__(self, filename: str):
        self.filename = filename

        # Validate file or create new
        if os.path.exists(filename):
            pass
        else:
            with open(filename, "w") as f:
                f.write("Date, {All columns of interest}, predicted value, buy signal\n")

    def _add_line(self, line: [str]):
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)

    # def log(self, data from model):
    #     pass

